"""
camera_stream.py  (v6)
FIXED: Windows MSMF error (-1072875772) — forces DirectShow on Windows.
FIXED: Hikvision RTSP delay — low-latency FFmpeg options + CAP_PROP_BUFFERSIZE=1
       + grab/retrieve drain loop so stale buffered frames are skipped.
       for webcams on Windows, with automatic fallback to default backend.
NEW:   BlockageDetector — detects camera blackout / physical obstruction and
       fires an alert via AlertSystem. Three independent signals are combined:
         1. Mean brightness  — very dark frame → lens covered / lights off
         2. Pixel variance   — near-zero variance → solid colour block/tape
         3. Edge density     — almost no edges → blurry cloth / spray paint
       All three must stay below threshold for `confirm_seconds` before the
       alert fires, reducing false positives from momentary dark scenes.
"""
import cv2, time, threading, json, platform, sys, numpy as np


def _open_webcam(index):
    """
    Open a webcam with the best available backend.
    On Windows: try DirectShow first (avoids MSMF grab errors),
                then fall back to MSMF, then default.
    On other OS: use default.
    """
    cap = None
    if platform.system() == 'Windows':
        # 1st choice: DirectShow — most reliable on Windows for USB/laptop cams
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f'[CAMERA] Opened webcam {index} via DirectShow (DSHOW)')
                return cap
            cap.release()

        # 2nd choice: MSMF (sometimes works on newer drivers)
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f'[CAMERA] Opened webcam {index} via MSMF')
                return cap
            cap.release()

        # 3rd choice: let OpenCV decide
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f'[CAMERA] Opened webcam {index} via default backend')
            return cap
        cap.release()
        return None
    else:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
        return None


# ---------------------------------------------------------------------------
# Camera Blockage / Blackout Detector
# ---------------------------------------------------------------------------

class BlockageDetector:
    """
    Analyses individual frames to decide whether the camera is physically
    blocked or blacked out.

    Detection logic (all three signals must agree to trigger):
      • brightness_thresh  — mean pixel intensity (0-255). A covered lens or
                             switched-off IR-cut is very dark (< ~15).
      • variance_thresh    — spatial variance of the greyscale frame. A cloth,
                             hand, or tape pressed on the lens produces a near-
                             uniform image (< ~20).
      • edge_density_thresh— fraction of pixels that are edges (Canny). A
                             clean scene has lots of edges; an obstruction has
                             almost none (< ~0.002 = 0.2 %).

    Any ONE signal can be low due to a legitimately dark/plain scene.
    Requiring ALL THREE to be simultaneously low makes false positives rare.

    `confirm_seconds` — how many consecutive seconds all signals must stay
    low before the blockage is declared (default 3 s). This avoids alerting
    on a person walking close to the camera momentarily.

    `cooldown_seconds` — minimum gap between successive alerts for the same
    camera so the portal is not spammed (default 120 s).
    """

    def __init__(self,
                 brightness_thresh: float  = 15.0,
                 variance_thresh:   float  = 20.0,
                 edge_density_thresh: float = 0.002,
                 confirm_seconds:   float  = 3.0,
                 cooldown_seconds:  float  = 120.0):
        self.brightness_thresh   = brightness_thresh
        self.variance_thresh     = variance_thresh
        self.edge_density_thresh = edge_density_thresh
        self.confirm_seconds     = confirm_seconds
        self.cooldown_seconds    = cooldown_seconds

        self._suspect_since: float | None = None   # time blockage first seen
        self._last_alert_at: float        = 0.0    # time last alert was fired
        self._blocked: bool               = False  # current declared state

    # ------------------------------------------------------------------
    def analyse(self, frame) -> dict:
        """
        Analyse one frame.

        Returns a dict:
          {
            'blocked':        bool,   # True = camera is blocked RIGHT NOW
            'just_triggered': bool,   # True = blockage was just newly declared
            'just_cleared':   bool,   # True = blockage just cleared
            'reason':         str,    # human-readable reason string
            'brightness':     float,
            'variance':       float,
            'edge_density':   float,
          }
        """
        result = {
            'blocked': False,
            'just_triggered': False,
            'just_cleared': False,
            'reason': '',
            'brightness': 0.0,
            'variance': 0.0,
            'edge_density': 0.0,
        }

        if frame is None:
            return result

        # --- Convert to greyscale once --------------------------------
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Signal 1: Mean brightness
        brightness = float(np.mean(grey))

        # Signal 2: Pixel variance (std-dev squared)
        variance = float(np.var(grey))

        # Signal 3: Edge density (fraction of edge pixels via Canny)
        edges      = cv2.Canny(grey, 30, 100)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        result['brightness']   = round(brightness,   2)
        result['variance']     = round(variance,     2)
        result['edge_density'] = round(edge_density, 5)

        # --- Determine reason string for alert message ----------------
        reasons = []
        dark     = brightness   < self.brightness_thresh
        uniform  = variance     < self.variance_thresh
        no_edges = edge_density < self.edge_density_thresh

        if dark:
            reasons.append(f'blackout (brightness={brightness:.1f})')
        if uniform:
            reasons.append(f'uniform image (variance={variance:.1f})')
        if no_edges:
            reasons.append(f'no edges (density={edge_density:.4f})')

        # All three signals must agree
        suspect_now = dark and uniform and no_edges

        now = time.time()

        if suspect_now:
            if self._suspect_since is None:
                self._suspect_since = now   # start timer

            elapsed = now - self._suspect_since
            result['reason'] = '; '.join(reasons)

            if elapsed >= self.confirm_seconds and not self._blocked:
                # Only fire if cooldown has passed
                if (now - self._last_alert_at) >= self.cooldown_seconds:
                    self._blocked         = True
                    result['blocked']     = True
                    result['just_triggered'] = True
                    self._last_alert_at   = now
                else:
                    self._blocked     = True
                    result['blocked'] = True
            else:
                result['blocked'] = self._blocked   # carry existing state

        else:
            # Signals look normal — reset suspect timer
            if self._blocked:
                result['just_cleared'] = True
            self._suspect_since = None
            self._blocked       = False
            result['blocked']   = False

        return result

    def reset(self):
        """Call this after reconnect / camera restart."""
        self._suspect_since = None
        self._last_alert_at = 0.0
        self._blocked       = False


# ---------------------------------------------------------------------------
class CameraStream:

    def __init__(self, source=None, label='Camera', rtsp_url=None,
                 max_retry=5, cam_id=None, config_path=None,
                 blockage_cfg: dict = None):
        if config_path and source is None:
            with open(config_path) as f:
                cfg = json.load(f)['camera']
            source    = cfg.get('source', 'webcam')
            rtsp_url  = cfg.get('rtsp_url', '')
            label     = cfg.get('label', 'Main Camera')
            max_retry = cfg.get('reconnect_attempts', 5)

        self.cam_id    = cam_id
        self.label     = label
        self.max_retry = max_retry
        self.cap       = None
        self.frame     = None
        self.running   = False
        self._lock     = threading.Lock()
        self._fail_count = 0
        self._MAX_FAILS  = 30   # consecutive read failures before reconnect

        # --- Blockage detector ----------------------------------------
        bcfg = blockage_cfg or {}
        self.blockage_detector = BlockageDetector(
            brightness_thresh    = bcfg.get('brightness_thresh',    15.0),
            variance_thresh      = bcfg.get('variance_thresh',      20.0),
            edge_density_thresh  = bcfg.get('edge_density_thresh',  0.002),
            confirm_seconds      = bcfg.get('confirm_seconds',      3.0),
            cooldown_seconds     = bcfg.get('cooldown_seconds',     120.0),
        )
        # Latest blockage status — read by portal.py / consumers
        self.blockage_status: dict = {
            'blocked': False, 'reason': '',
            'brightness': 0.0, 'variance': 0.0, 'edge_density': 0.0,
        }
        # Optional callback: fn(camera_label, reason, status_dict)
        self.on_blocked_callback  = None
        self.on_cleared_callback  = None

        if source == 'webcam' or source == 0 or source == '0':
            self._src  = 0
            self._type = 'webcam'
        elif source == 'rtsp' or (isinstance(source, str) and source.startswith('rtsp')):
            self._src  = rtsp_url if source == 'rtsp' else source
            self._type = 'rtsp'
        elif isinstance(source, int):
            self._src  = source
            self._type = 'webcam'
        else:
            try:
                self._src  = int(source)
                self._type = 'webcam'
            except (ValueError, TypeError):
                self._src  = source
                self._type = 'rtsp'

    def connect(self):
        for attempt in range(self.max_retry):
            print(f'[CAMERA] Connecting to {self.label} (attempt {attempt+1}/{self.max_retry})...')

            if self._type == 'rtsp':
                # FIX: Pass FFmpeg options to minimise internal buffering.
                # probesize / analyzeduration reduce the startup negotiation delay.
                # reorder_queue_size=0 disables the reorder buffer entirely.
                # flags=low_delay tells FFmpeg to skip frame reordering.
                rtsp_opts = (
                    "rtsp_transport;tcp"          # TCP = more reliable than UDP
                    "|buffer_size;65536"           # small socket buffer
                    "|max_delay;500000"            # max 0.5 s decoder delay
                    "|reorder_queue_size;0"        # no reorder buffer
                    "|probesize;32"                # fast probe
                    "|analyzeduration;0"           # skip duration analysis
                    "|flags;low_delay"             # low-latency decode flag
                    "|fflags;nobuffer+discardcorrupt"
                )
                self.cap = cv2.VideoCapture(
                    f"{self._src}?{rtsp_opts}" if "?" not in str(self._src)
                    else self._src,
                    cv2.CAP_FFMPEG
                )
                if self.cap.isOpened():
                    # FIX: Set OpenCV's own internal ring-buffer to 1 frame.
                    # Default is ~100 frames — that alone causes seconds of lag.
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Warm up — drain any frames already queued during connect
                    for _ in range(5):
                        self.cap.grab()
                    print(f'[CAMERA] Connected — {self.label} (RTSP)')
                    self._fail_count = 0
                    return True
            else:
                # Webcam — use DSHOW-first logic
                self.cap = _open_webcam(int(self._src) if isinstance(self._src, (int, str)) else 0)
                if self.cap is not None:
                    print(f'[CAMERA] Connected — {self.label} (webcam index {self._src})')
                    self._fail_count = 0
                    return True

            print(f'[CAMERA] Failed attempt {attempt+1}, retrying in 2s...')
            time.sleep(2)

        raise ConnectionError(f'[CAMERA] Cannot open {self.label} after {self.max_retry} attempts.')

    def _loop(self):
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue

            if self._type == 'rtsp':
                # FIX: For RTSP, use grab() in a tight loop to drain the decoder
                # queue, then retrieve() only the very last grabbed frame.
                # This skips stale buffered frames instead of decoding every one.
                grabbed = False
                for _ in range(4):          # drain up to 4 queued frames
                    ok = self.cap.grab()
                    if ok:
                        grabbed = True
                    else:
                        break
                if grabbed:
                    ret, frame = self.cap.retrieve()
                else:
                    ret, frame = False, None
            else:
                # Webcam — simple read() is fine
                ret, frame = self.cap.read()

            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
                self._fail_count = 0

                # --- Blockage detection (runs on every good frame) ----
                status = self.blockage_detector.analyse(frame)
                self.blockage_status = status

                if status['just_triggered']:
                    print(f'[BLOCKAGE] ⚠  {self.label}: camera BLOCKED — {status["reason"]}')
                    if self.on_blocked_callback:
                        try:
                            self.on_blocked_callback(self.label, status['reason'], status)
                        except Exception as cb_err:
                            print(f'[BLOCKAGE] callback error: {cb_err}')

                elif status['just_cleared']:
                    print(f'[BLOCKAGE] ✓  {self.label}: camera CLEAR again.')
                    if self.on_cleared_callback:
                        try:
                            self.on_cleared_callback(self.label, status)
                        except Exception as cb_err:
                            print(f'[BLOCKAGE] cleared-callback error: {cb_err}')
            else:
                self._fail_count += 1
                if self._fail_count >= self._MAX_FAILS:
                    print(f'[CAMERA] {self.label}: too many read failures, reconnecting...')
                    try:
                        if self.cap:
                            self.cap.release()
                        self.connect()
                        self.blockage_detector.reset()   # fresh state after reconnect
                    except ConnectionError:
                        print(f'[CAMERA] {self.label}: reconnect failed, stopping.')
                        self.running = False
                time.sleep(0.05)

    def start(self):
        self.connect()
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True, name=f'cam-{self.label}')
        t.start()

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def is_blocked(self) -> bool:
        """Quick check — True if the camera is currently declared blocked."""
        return self.blockage_status.get('blocked', False)

    def get_source_label(self):
        src_type = 'RTSP' if self._type == 'rtsp' else 'Webcam'
        return f'{self.label} ({src_type})'

    def stop(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
            self.cap = None


def load_cameras_from_config(config_path='config.json'):
    """Return list of CameraStream objects from the cameras list in config."""
    with open(config_path) as f:
        cfg = json.load(f)

    cameras_cfg  = cfg.get('cameras', [])
    blockage_cfg = cfg.get('blockage_detection', {})   # global blockage settings

    if not cameras_cfg:
        # Fallback: use legacy single [camera] block
        print('[CAMERA] No cameras list in config — using legacy [camera] block.')
        return [CameraStream(config_path=config_path, cam_id='cam0',
                             blockage_cfg=blockage_cfg)]

    streams = []
    for i, c in enumerate(cameras_cfg):
        # Per-camera blockage overrides merged on top of global settings
        merged_bcfg = {**blockage_cfg, **c.get('blockage_detection', {})}
        cam = CameraStream(
            source       = c.get('source', 'webcam'),
            label        = c.get('label', f'Camera {i+1}'),
            rtsp_url     = c.get('rtsp_url', ''),
            max_retry    = c.get('reconnect_attempts', 5),
            cam_id       = c.get('id', f'cam{i}'),
            blockage_cfg = merged_bcfg,
        )
        streams.append(cam)
    return streams