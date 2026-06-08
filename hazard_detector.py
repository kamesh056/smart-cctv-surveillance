"""
hazard_detector.py  (v2 — NeuralGuard fixed edition)

Fixes vs v1:
  - FIX DEADLOCK: detect() acquired both `outer` (infer_lock) AND self._lock.
    If infer_lock was the same object as self._lock this deadlocked. Now uses
    ONLY the provided infer_lock OR self._lock — never both.
  - FIX FALSE FLOOD: flood_area_pct default raised from 0.07 → 0.25 in
    HazardDetector.__init__. The config.json value (0.30) is used when
    provided. Low threshold caused constant false FLOOD alerts from indoor
    webcams seeing blue walls/uniforms.
  - FIX: flood_sat_low raised to 80 (from 60) to avoid triggering on
    low-saturation grey/white backgrounds.
  - FIX: fire/smoke YOLO imgsz now adaptive (480 if short side ≤ 480).
  - FIX: get_hazard_detector() respects config flood_area_pct correctly.
"""

import cv2
import time
import logging
import threading
import numpy as np

log = logging.getLogger(__name__)

CLASS_FIRE  = 0
CLASS_SMOKE = 1

COLOR_FIRE  = (0,   50,  255)
COLOR_SMOKE = (130, 130, 130)
COLOR_FLOOD = (200, 160,   0)


class HazardDetector:
    """
    Detects fire, smoke, and flood.
    Thread-safe — one shared instance across all camera threads.
    """

    def __init__(self,
                 fire_model:      str   = 'yolov8n-fire.pt',
                 confidence:      float = 0.50,
                 process_every:   int   = 10,
                 cooldown_seconds: float = 120.0,
                 flood_hue_low:   int   = 90,
                 flood_hue_high:  int   = 130,
                 flood_sat_low:   int   = 80,
                 flood_area_pct:  float = 0.25):  # FIX: was 0.07, way too sensitive

        self.conf           = confidence
        self.process_every  = process_every
        self.cooldown       = cooldown_seconds
        self.flood_hue_low  = flood_hue_low
        self.flood_hue_high = flood_hue_high
        self.flood_sat_low  = flood_sat_low
        self.flood_area_pct = flood_area_pct

        self._lock       = threading.Lock()
        self._last_alert = {}   # {cam_id: {hazard_type: last_alert_time}}
        self.model       = None

        try:
            from ultralytics import YOLO
            self.model = YOLO(fire_model)
            log.info(f'[HAZARD] Fire/smoke model loaded: {fire_model}')

            # Warm-up
            dummy = np.zeros((320, 320, 3), dtype='uint8')
            self.model(dummy, verbose=False)
            log.info('[HAZARD] Warm-up complete.')
        except Exception as e:
            log.warning(f'[HAZARD] Fire model load failed ({e}) — fire/smoke detection disabled.')

    @staticmethod
    def _infer_imgsz(frame) -> int:
        h, w = frame.shape[:2]
        return 480 if min(h, w) <= 480 else 640

    def detect(self, frame, infer_lock=None) -> list:
        """
        Run fire/smoke YOLO + flood colour analysis on one frame.

        FIX: Uses ONLY the provided infer_lock OR self._lock — never both.
        Eliminates the deadlock from v1 where both were acquired.
        """
        if frame is None or frame.size == 0:
            return []

        detections = []

        # Fire / smoke (YOLO)
        if self.model is not None:
            try:
                imgsz = self._infer_imgsz(frame)
                # FIX: use ONLY one lock — not both outer and self._lock
                lock = infer_lock if infer_lock is not None else self._lock
                with lock:
                    results = self.model(frame, conf=self.conf,
                                         verbose=False, imgsz=imgsz)
                for r in results:
                    for box in r.boxes:
                        cls  = int(box.cls[0])
                        conf = round(float(box.conf[0]), 3)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        h_type = 'fire' if cls == CLASS_FIRE else 'smoke'
                        area_px = max((x2 - x1) * (y2 - y1), 1)
                        frame_px = max(frame.shape[0] * frame.shape[1], 1)
                        detections.append({
                            'type':       h_type,
                            'confidence': conf,
                            'box':        (x1, y1, x2, y2),
                            'area_pct':   round(area_px / frame_px, 4),
                        })
            except Exception as e:
                log.warning(f'[HAZARD] YOLO fire detect error: {e}')

        # Flood / water (HSV colour analysis)
        try:
            flood_det = self._detect_flood(frame)
            if flood_det:
                detections.append(flood_det)
        except Exception as e:
            log.warning(f'[HAZARD] Flood detect error: {e}')

        return detections

    def _detect_flood(self, frame) -> dict | None:
        """Detect flood via HSV blue-range analysis."""
        h, w      = frame.shape[:2]
        total_px  = h * w

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.flood_hue_low,  self.flood_sat_low, 40]),
            np.array([self.flood_hue_high,  255,               255]),
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        area_pct = float(np.count_nonzero(mask)) / total_px

        if area_pct < self.flood_area_pct:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return {
            'type':       'flood',
            'confidence': round(min(area_pct / self.flood_area_pct, 1.0), 3),
            'box':        (x, y, x + bw, y + bh),
            'area_pct':   round(area_pct, 4),
        }

    def draw(self, frame, detections: list):
        if not detections:
            return frame

        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']
            h_type = d['type']

            if h_type == 'fire':
                color = COLOR_FIRE
                label = f"FIRE {d['confidence']:.0%}"
            elif h_type == 'smoke':
                color = COLOR_SMOKE
                label = f"SMOKE {d['confidence']:.0%}"
            else:
                color = COLOR_FLOOD
                label = f"FLOOD {d['area_pct'] * 100:.1f}% area"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1),
                          color, cv2.FILLED)
            cv2.putText(out, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

            if h_type == 'flood':
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
                out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)

        return out

    def maybe_alert(self, detection: dict, camera_label: str,
                    alerts_store, cam_id: str):
        h_type  = detection['type']
        now     = time.time()
        cam_log = self._last_alert.setdefault(cam_id, {})

        if now - cam_log.get(h_type, 0) < self.cooldown:
            return

        cam_log[h_type] = now
        log.warning(f'[HAZARD] {camera_label}: {h_type.upper()} detected '
                    f'(conf={detection["confidence"]:.0%})')

        try:
            alerts_store.AlertSystem().send_hazard_alert(
                camera  = camera_label,
                hazard  = h_type,
                details = (f"Confidence: {detection['confidence']:.0%}  "
                           f"Area: {detection.get('area_pct', 0) * 100:.1f}%"),
            )
        except AttributeError:
            log.warning('[HAZARD] alert_system.py missing send_hazard_alert()')
        except Exception as e:
            log.warning(f'[HAZARD] alert error: {e}')


# ── Singleton ──────────────────────────────────────────────────────────────

_hazard_instance = None
_hazard_lock     = threading.Lock()


def get_hazard_detector(cfg: dict = None) -> 'HazardDetector | None':
    """
    Returns a shared HazardDetector instance.
    Returns None if hazard.enabled = false in config.
    """
    global _hazard_instance
    with _hazard_lock:
        if _hazard_instance is not None:
            return _hazard_instance

        cfg = cfg or {}
        if not cfg.get('enabled', True):
            log.info('[HAZARD] Disabled in config.')
            return None

        try:
            _hazard_instance = HazardDetector(
                fire_model       = cfg.get('fire_model',        'yolov8n-fire.pt'),
                confidence       = cfg.get('confidence',         0.50),
                process_every    = cfg.get('process_every',      10),
                cooldown_seconds = cfg.get('cooldown_seconds',   120.0),
                flood_hue_low    = cfg.get('flood_hue_low',      90),
                flood_hue_high   = cfg.get('flood_hue_high',     130),
                flood_sat_low    = cfg.get('flood_sat_low',      80),
                flood_area_pct   = cfg.get('flood_area_pct',     0.25),
            )
        except Exception as e:
            log.error(f'[HAZARD] Failed to initialise: {e}')
            return None

        return _hazard_instance