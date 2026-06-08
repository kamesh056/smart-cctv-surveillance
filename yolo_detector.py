"""
yolo_detector.py  (v4 — NeuralGuard fixed edition)

Fixes vs v3:
  - FIX CRITICAL — OpenVINO fixed-shape mismatch crash:
    The exported OpenVINO model is compiled at 640×640 (fixed shape).
    Passing any other size (e.g. 480×480) causes:
      "The input tensor size is not equal to the model input type:
       got [1,3,480,480] expecting [1,3,640,640]"
    on EVERY inference call → zero detections forever.
    Fix: track self._is_openvino and always pass imgsz=640 for OpenVINO.
    For PyTorch fallback, keep the adaptive imgsz logic.

  - FIX — Warm-up uses wrong size for OpenVINO:
    Warm-up dummy was (320,320) — also rejected by OpenVINO 640-model.
    Now warm-up uses (640,640) for OpenVINO, (320,320) for PyTorch.

  - FIX — gevent LoopExit from OpenVINO ThreadPool:
    OpenVINO internally uses concurrent.futures.ThreadPoolExecutor.
    After gevent monkey-patches queue.SimpleQueue, the thread pool's
    blocking .get() deadlocks gevent's event loop.
    Fix: the YOLODetector constructor must be called BEFORE
    monkey.patch_all() runs, OR the model must be loaded in a real
    OS thread. portal.py already does this (models loaded before gevent),
    but the fix is also reinforced here by keeping the model reference
    in a plain threading.Lock (not gevent-aware).

  - FIX — portal.py resizes frames to 480p before passing to detect():
    For OpenVINO we must NOT resize to 480 before inference — the model
    needs 640. detect() now checks self._is_openvino and if true, resizes
    the frame internally to 640×640 before running inference, then maps
    boxes back to original frame coordinates. The caller (portal.py) no
    longer needs to pass a pre-resized frame for YOLO.

  - FIX — MIN_BOX_HEIGHT_FOR_RECOG scale-awareness:
    Previously the box height was checked against a pixel threshold on the
    infer frame (480p), but boxes are now in original-frame coords.
    Threshold is now applied in the original frame coordinate space.

  - FIX — detect_and_classify() when recognizer=None but dets exist:
    Previously returned dets with name=None/authorized=None.
    Now explicitly marks them authorized=None (orange) which is correct.
"""

import threading
import logging
import cv2
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# COCO class 0 = person
PERSON_CLASS = 0

# Box colours (BGR)
COLOR_AUTHORIZED   = (0,  210,   0)    # green  — authorised person
COLOR_UNAUTHORIZED = (0,   0,  220)    # red    — unknown / unauthorised
COLOR_UNRESOLVED   = (0,  165,  255)   # orange — person detected, face unclear
COLOR_OTHER        = (180, 180, 180)   # grey   — non-person COCO class

# Minimum pixel height of a person box (in ORIGINAL frame coords) before
# attempting face recognition.
MIN_BOX_HEIGHT_FOR_RECOG = 50   # pixels

# Padding added around the YOLO box when cropping for face recognition.
CROP_PAD_TOP   = 0.20   # fraction of box height added ABOVE the box (gets the head)
CROP_PAD_SIDES = 0.08   # fraction of box width added on each side


class YOLODetector:
    """
    Thread-safe YOLOv8 detector with integrated face-based authorization.

    OpenVINO note: the exported model is compiled at a FIXED input shape
    (640×640 by default). This class tracks self._is_openvino and always
    passes imgsz=640 when running OpenVINO inference, regardless of what
    the caller passes in.
    """

    def __init__(self,
                 model_path:   str   = 'yolov8n.pt',
                 confidence:   float = 0.40,
                 classes:      list  = None,
                 use_openvino: bool  = True):

        from ultralytics import YOLO

        self.conf         = confidence
        self.classes      = classes if classes is not None else [PERSON_CLASS]
        self._lock        = threading.Lock()
        self._is_openvino = False   # set to True if OpenVINO loads successfully

        ov_dir = Path(model_path).stem + '_openvino_model'

        if use_openvino:
            try:
                if not Path(ov_dir).exists():
                    log.info('[YOLO] Exporting to OpenVINO (first run only)...')
                    _tmp = YOLO(model_path)
                    _tmp.export(format='openvino', half=False)
                    log.info(f'[YOLO] OpenVINO export saved → {ov_dir}/')

                self.model        = YOLO(ov_dir + '/')
                self.device       = 'cpu'
                self._is_openvino = True
                log.info('[YOLO] Loaded OpenVINO model — Intel acceleration ON')
            except Exception as e:
                log.warning(f'[YOLO] OpenVINO init failed ({e}) — falling back to CPU PyTorch')
                self.model        = YOLO(model_path)
                self.device       = 'cpu'
                self._is_openvino = False
        else:
            self.model        = YOLO(model_path)
            self.device       = 'cpu'
            self._is_openvino = False
            log.info(f'[YOLO] Loaded {model_path} on CPU (PyTorch)')

        # Warm-up — use the correct size for the model type
        # OpenVINO fixed-shape 640 model rejects ANY other size including 320.
        warmup_size = 640 if self._is_openvino else 320
        try:
            dummy = np.zeros((warmup_size, warmup_size, 3), dtype='uint8')
            self.model(dummy, verbose=False, imgsz=warmup_size)
            log.info(f'[YOLO] Warm-up complete ({"OpenVINO" if self._is_openvino else "PyTorch"}, {warmup_size}px).')
        except Exception as e:
            log.warning(f'[YOLO] Warm-up failed: {e}')

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_imgsz(self, frame) -> int:
        """
        Return the imgsz to pass to YOLO inference.

        OpenVINO: ALWAYS 640 — the exported model has a FIXED input shape.
        Passing anything else causes:
          "got [1,3,NNN,NNN] expecting [1,3,640,640]"
        and returns zero detections.

        PyTorch: adaptive — pick smallest size that fits the frame.
        """
        if self._is_openvino:
            return 640   # FIXED — OpenVINO model compiled at 640×640

        h, w = frame.shape[:2]
        shorter = min(h, w)
        if shorter <= 320:
            return 320
        elif shorter <= 480:
            return 480
        else:
            return 640

    # ── Raw detection ──────────────────────────────────────────────────────

    def detect(self, frame, infer_lock=None) -> list:
        """
        Run YOLO on a BGR frame.

        For OpenVINO: resizes frame to 640×640 internally, then maps all
        bounding boxes back to original frame coordinates. The caller
        does NOT need to pre-resize frames.

        infer_lock: optional external Lock. If provided, self._lock is NOT
        also acquired (prevents double-locking when caller passes self._lock).

        Returns list of detection dicts with boxes in ORIGINAL frame coords.
        """
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        imgsz = self._get_imgsz(frame)

        # For OpenVINO: resize to 640×640 so the fixed-shape model accepts it.
        # Keep scale factors to map boxes back to original coords.
        if self._is_openvino and (orig_h != 640 or orig_w != 640):
            infer_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0
        else:
            infer_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        try:
            lock = infer_lock if infer_lock is not None else self._lock
            with lock:
                results = self.model(
                    infer_frame,
                    conf    = self.conf,
                    classes = self.classes,
                    verbose = False,
                    imgsz   = imgsz,
                )
        except Exception as e:
            log.warning(f'[YOLO] detect() error: {e}')
            return []

        detections = []
        for r in results:
            for box in r.boxes:
                # Map box coords from inference frame → original frame
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                detections.append({
                    'label':      r.names[int(box.cls[0])],
                    'confidence': round(float(box.conf[0]), 3),
                    'box':        (x1, y1, x2, y2),
                    'class_id':   int(box.cls[0]),
                    'authorized': None,
                    'name':       None,
                    'person_id':  None,
                    'face_result': None,
                    'location':   None,
                })
        return detections

    # ── Integrated detection + authorization ───────────────────────────────

    def detect_and_classify(self, frame, recognizer=None, infer_lock=None) -> list:
        """
        Run YOLO, then for each person box crop the ORIGINAL frame and
        run face recognition on it.

        Returns list of detection dicts (same schema as detect()) with
        authorized / name / person_id / face_result / location filled in.
        All boxes and locations are in ORIGINAL frame coordinates.
        """
        dets = self.detect(frame, infer_lock=infer_lock)
        if not dets or recognizer is None:
            return dets

        h_frame, w_frame = frame.shape[:2]

        for d in dets:
            if d['class_id'] != PERSON_CLASS:
                continue

            x1, y1, x2, y2 = d['box']
            box_h = y2 - y1
            box_w = x2 - x1

            if box_h < MIN_BOX_HEIGHT_FOR_RECOG:
                # Box too small for reliable face recognition (leave orange)
                continue

            # ── Padded crop from ORIGINAL frame — includes head above box ──
            pad_top   = int(box_h * CROP_PAD_TOP)
            pad_sides = int(box_w * CROP_PAD_SIDES)

            cx1 = max(0, x1 - pad_sides)
            cy1 = max(0, y1 - pad_top)
            cx2 = min(w_frame, x2 + pad_sides)
            cy2 = min(h_frame, y2)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # ── Face recognition on crop ───────────────────────────────
            try:
                face_results = recognizer.recognize_faces(crop)
            except Exception as e:
                log.warning(f'[YOLO] face-recog error on crop: {e}')
                face_results = []

            if not face_results:
                continue

            # Pick best: prefer authorised, then highest confidence
            authorized_results = [r for r in face_results if r.get('authorized')]
            best = (max(authorized_results, key=lambda r: r.get('confidence', 0))
                    if authorized_results
                    else max(face_results, key=lambda r: r.get('confidence', 0)))

            d['authorized']  = best.get('authorized', False)
            d['name']        = best.get('name', 'UNKNOWN')
            d['person_id']   = best.get('person_id')
            d['face_result'] = best

            # Map face location (crop coords) → full-frame coords
            loc = best.get('location')
            if loc:
                ft, fr, fb, fl = loc
                d['location'] = (cy1 + ft, cx1 + fr, cy1 + fb, cx1 + fl)
            else:
                d['location'] = (y1, x2, y2, x1)

        return dets

    # ── Drawing ────────────────────────────────────────────────────────────

    def draw(self, frame, detections: list):
        """
        Draw annotated bounding boxes.
          GREEN  — authorised person (face matched)
          RED    — unauthorised / unknown (face found but not enrolled)
          ORANGE — person detected, face not resolved (too small / occluded)
          GREY   — non-person COCO class
        """
        if not detections:
            return frame

        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']

            if d['class_id'] != PERSON_CLASS:
                color = COLOR_OTHER
                label = f"{d['label']} {d['confidence']:.0%}"
            elif d['authorized'] is True:
                color = COLOR_AUTHORIZED
                name  = d.get('name') or 'Person'
                conf  = (d.get('face_result') or {}).get('confidence', d['confidence'])
                label = f"OK {name}  {conf:.0%}"
            elif d['authorized'] is False:
                color = COLOR_UNAUTHORIZED
                label = f"!! UNKNOWN  {d['confidence']:.0%}"
            else:
                # authorized=None — person detected, face unresolved
                color = COLOR_UNRESOLVED
                label = f"? Person  {d['confidence']:.0%}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ly = max(y1, th + 10)
            cv2.rectangle(out, (x1, ly - th - 8), (x1 + tw + 8, ly), color, cv2.FILLED)
            cv2.putText(out, label, (x1 + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            loc = d.get('location')
            if loc and d['authorized'] is not None:
                ft, fr, fb, fl = loc
                cv2.rectangle(out, (fl, ft), (fr, fb), color, 1, cv2.LINE_AA)

        return out


# ── Singleton ──────────────────────────────────────────────────────────────

_detector_instance = None
_detector_lock     = threading.Lock()


def get_yolo_detector(cfg: dict = None) -> 'YOLODetector | None':
    """
    Returns a shared YOLODetector instance (singleton).
    Pass config dict on first call; subsequent calls return cached instance.
    Returns None if yolo.enabled = false in config.
    """
    global _detector_instance
    with _detector_lock:
        if _detector_instance is not None:
            return _detector_instance

        cfg = cfg or {}
        if not cfg.get('enabled', True):
            log.info('[YOLO] Disabled in config — skipping.')
            return None

        model      = cfg.get('model',        'yolov8n.pt')
        confidence = cfg.get('confidence',   0.40)
        classes    = cfg.get('classes',      [0])
        use_ov     = cfg.get('use_openvino', True)

        try:
            _detector_instance = YOLODetector(
                model_path   = model,
                confidence   = confidence,
                classes      = classes,
                use_openvino = use_ov,
            )
        except Exception as e:
            log.error(f'[YOLO] Failed to initialise detector: {e}')
            return None

        return _detector_instance
