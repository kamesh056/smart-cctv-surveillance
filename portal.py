"""
portal.py  (v15 — NeuralGuard fully fixed edition)

Fixes applied in this version
──────────────────────────────

FIX 1  gevent LoopExit crash
    monkey.patch_all(thread=False) was causing the concurrent.futures
    ThreadPoolExecutor (used internally by gevent's Hub) to call
    queue.SimpleQueue.get() which blocks forever under gevent's event loop.
    Fix: monkey.patch_all(thread=False, subprocess=False)
    Also: _GLOBAL_RECOG_Q is created BEFORE monkey-patching so it uses the
    real stdlib queue, not the gevent-patched one.

FIX 2  Authorised-person cache not invalidated on delete
    After deleting a person the old face_encodings.pkl was still loaded in
    _recognizer.  _run_encoder_background() now forces _recognizer to None
    immediately (not just _recognizer_mtime=0) so the very next frame does
    a full reload from the freshly written pkl.

FIX 3  Newly enrolled person not recognised
    Same root cause as FIX 2.  After video_face_encoder.py finishes the
    recognition singleton was not reloaded until the mtime changed.  Now
    _recognizer is set to None immediately after the encoder completes AND
    _recognizer_mtime is reset, guaranteeing a fresh load on the next frame.

FIX 4  Sub-threshold log spam (50+ lines/second)
    face_recognizer.py v7 throttles the rejection log to at most 1 line per
    30 seconds.  This eliminates the I/O bottleneck that was slowing the
    entire system down.

FIX 5  Detection thread crash → never restarts
    The detection worker loop is now wrapped in an outer retry loop.
    If the inner loop crashes the thread sleeps 5 seconds and restarts,
    clearing _started_cams so ensure_detection_thread() can be called again.

FIX 6  Global _cv2_read_lock blocking both cameras
    Webcam reads from camera 1 were blocking camera 2 because both shared
    ONE global lock.  Now each webcam camera gets its own per-camera lock.

FIX 7  No sleep in detection loop → 100% CPU on webcam
    When consecutive_fails > 0 and cam_type == 'webcam', the loop was
    spinning without sleeping.  Now always sleeps at least 33 ms between
    reads to cap at ~30 fps even when reads are fast.

FIX 8  imencode on detection thread blocking next frame
    JPEG encoding is now done after drawing and before the next grab, not
    inside the heavy inference block. Moved the encode to the end of the
    loop, decoupled from inference timing.

FIX 9  Full-frame fallback running every 20 frames even when all resolved
    Now skips the full-frame subprocess IPC call when YOLO has already
    resolved all visible persons as authorised.

FIX 10  config.json: webcam hazard_enabled=false, process_every tuned
    See config.json changes.
"""

# ── MUST be set before ANY cv2 import ─────────────────────────────────────
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|timeout;15000000|max_delay;500000'
    '|stimeout;15000000|allowed_media_types;video'
)
os.environ['OPENCV_LOG_LEVEL']             = 'ERROR'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# ── Create the real stdlib queue BEFORE gevent monkey-patch ───────────────
# gevent patches queue.Queue/SimpleQueue in monkey.patch_all().
# The recog dispatcher uses blocking queue.get() which deadlocks under
# gevent's event loop.  Creating the queue here (before patching) preserves
# the real threading-based implementation.
import queue as _real_queue
_GLOBAL_RECOG_Q_IMPL = _real_queue.Queue(maxsize=4)

import json, platform, secrets, shutil, subprocess, sys
import time, threading, logging
from functools import wraps
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, send_from_directory,
                   Response, stream_with_context, session)
from werkzeug.utils import secure_filename

import db_mysql as db
import alert_system as alerts_store
from yolo_detector import get_yolo_detector
from hazard_detector import get_hazard_detector

# ── Logging ────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)

import io as _io
_safe_stdout = (
    _io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stdout, 'buffer') else sys.stdout
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log', encoding='utf-8'),
        logging.StreamHandler(_safe_stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)

with open('config.json') as _f:
    _startup_cfg = json.load(_f)

app.secret_key                    = _startup_cfg['portal']['secret_key']
app.permanent_session_lifetime    = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH']  = 100 * 1024 * 1024

PERSONS_DIR   = Path('authorized_persons')
SNAPSHOTS_DIR = Path('snapshots')
CONFIG_FILE   = Path('config.json')
ALLOWED_VIDEO = {'mp4', 'mov', 'avi', 'webm', 'mkv'}

# ── Self-contained Add Person page HTML ───────────────────────────────────
ADD_PERSON_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enroll Person — NeuralGuard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f1a;color:#e0e0e0;font-family:'Segoe UI',sans-serif;min-height:100vh}
.topbar{background:#16162a;border-bottom:1px solid #2a2a3a;padding:.85rem 1.5rem;
  display:flex;align-items:center;justify-content:space-between}
.topbar h1{font-size:1rem;color:#5b8dee;letter-spacing:.08em;font-weight:700}
.back{color:#aaa;text-decoration:none;font-size:.85rem;padding:.4rem .9rem;
  border:1px solid #333;border-radius:6px;transition:border-color .2s}
.back:hover{border-color:#5b8dee;color:#fff}
.wrap{max-width:1000px;margin:2rem auto;padding:0 1.2rem}
h2{font-size:1.15rem;margin-bottom:1.5rem;color:#fff}
.tabs{display:flex;border-bottom:2px solid #2a2a3a;margin-bottom:0}
.tab{padding:.75rem 1.8rem;background:transparent;border:none;color:#888;
  font-size:.9rem;cursor:pointer;border-bottom:3px solid transparent;
  margin-bottom:-2px;transition:color .2s,border-color .2s}
.tab.active{color:#5b8dee;border-bottom-color:#5b8dee}
.card{background:#16162a;border:1px solid #2a2a3a;border-radius:10px;overflow:hidden}
.pane{padding:1.8rem;display:none}
.pane.active{display:block}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:2rem;align-items:start}
@media(max-width:680px){.layout{grid-template-columns:1fr}.cam-side{order:-1}}
.fg{display:flex;flex-direction:column;gap:.35rem;margin-bottom:1rem}
.fg label{font-size:.75rem;color:#aaa;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.fg input{background:#1a1a2a;border:1px solid #2a2a3a;border-radius:6px;
  color:#fff;padding:.6rem .8rem;font-size:.9rem;outline:none;transition:border-color .2s;width:100%}
.fg input:focus{border-color:#5b8dee}
.btn{display:inline-block;padding:.55rem 1.2rem;border-radius:7px;border:none;
  font-size:.88rem;font-weight:600;cursor:pointer;transition:opacity .2s;text-decoration:none}
.btn:hover{opacity:.85}
.btn-p{background:#5b8dee;color:#fff}
.btn-d{background:#e05555;color:#fff}
.btn-s{background:#3cb371;color:#fff}
.btn-g{background:#2a2a3a;color:#ccc;border:1px solid #3a3a4a}
.status{margin-top:.9rem;font-size:.85rem;color:#aaa;min-height:1.2em}
.rec-wrap{display:none;margin-top:1rem}
.cam-box{position:relative;border-radius:8px;overflow:hidden;background:#000}
.cam-box img{width:100%;display:block;border-radius:8px}
.rec-dot{display:none;position:absolute;top:10px;right:10px;background:#e00;
  color:#fff;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:700;
  animation:blink 1s infinite}
.hint{color:#555;font-size:.76rem;margin-top:.4rem;text-align:center}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
video{width:100%;border-radius:8px;max-height:200px;background:#000}
.flash{padding:.7rem 1rem;border-radius:7px;margin-bottom:1.2rem;font-size:.88rem}
.flash.success{background:#1a3a1a;color:#7dea7d;border:1px solid #2a5a2a}
.flash.error{background:#3a1a1a;color:#ea7d7d;border:1px solid #5a2a2a}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem 1.5rem}
@media(max-width:600px){.form-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="topbar">
  <h1>NeuralGuard</h1>
  <a href="__PERSONS_URL__" class="back">&larr; Back to Persons</a>
</div>
<div class="wrap">
  <h2>Enroll New Person</h2>
  <div class="card">
    <div class="tabs">
      <button class="tab active" id="tab-record" onclick="switchTab('record')">&#127909; Record Webcam</button>
      <button class="tab"        id="tab-upload" onclick="switchTab('upload')">&#128193; Upload Video</button>
    </div>
    <div class="pane active" id="pane-record">
      <div class="layout">
        <div>
          <div class="fg"><label>Full Name *</label>
            <input id="rn" placeholder="e.g. Ayush Sharma"></div>
          <div class="fg"><label>Role</label>
            <input id="rr" placeholder="e.g. Engineer"></div>
          <div class="fg"><label>Contact</label>
            <input id="rc" placeholder="Phone or email"></div>
          <div class="fg"><label>Department</label>
            <input id="rd" placeholder="e.g. Operations"></div>
          <div style="margin-top:1rem;display:flex;gap:.6rem;flex-wrap:wrap">
            <button class="btn btn-p" id="btn-start" onclick="startRec()">&#9210; Start Recording</button>
            <button class="btn btn-d" id="btn-stop"  onclick="stopRec()" style="display:none">&#9209; Stop</button>
          </div>
          <div class="status" id="status"></div>
          <div class="rec-wrap" id="rec-wrap">
            <p style="color:#7dea7d;margin-bottom:.6rem;font-size:.85rem">&#10003; Done! Review then enroll.</p>
            <video id="preview" controls></video>
            <div style="margin-top:.8rem;display:flex;gap:.6rem">
              <button class="btn btn-s" onclick="submitRec()">&#10004; Enroll Person</button>
              <button class="btn btn-g" onclick="resetRec()">&#8635; Re-record</button>
            </div>
          </div>
        </div>
        <div class="cam-side">
          <div class="cam-box">
            <img id="live-img" src="__FEED_URL__" alt="Live feed">
            <div class="rec-dot" id="rec-dot">&#9679; REC</div>
          </div>
          <p class="hint">Face the camera &mdash; tilt slightly left, right, up, down</p>
        </div>
      </div>
    </div>
    <div class="pane" id="pane-upload">
      <form method="POST" action="__ADD_URL__" enctype="multipart/form-data">
        <div class="form-grid">
          <div class="fg"><label>Full Name *</label>
            <input type="text" name="name" placeholder="e.g. Ayush Sharma" required></div>
          <div class="fg"><label>Role</label>
            <input type="text" name="role" placeholder="e.g. Engineer"></div>
          <div class="fg"><label>Contact</label>
            <input type="text" name="contact" placeholder="Phone or email"></div>
          <div class="fg"><label>Department</label>
            <input type="text" name="department" placeholder="e.g. Operations"></div>
          <div class="fg"><label>Video File *</label>
            <input type="file" name="video" accept="video/*" required>
            <span style="color:#555;font-size:.75rem">mp4 / mov / avi / webm / mkv</span></div>
        </div>
        <div style="margin-top:1.4rem;display:flex;gap:.7rem">
          <button type="submit" class="btn btn-s">&#128229; Enroll Person</button>
          <a href="__PERSONS_URL__" class="btn btn-g">Cancel</a>
        </div>
      </form>
    </div>
  </div>
</div>
<script>
function switchTab(t) {
  ['record','upload'].forEach(function(id) {
    document.getElementById('pane-' + id).classList.toggle('active', id === t);
    document.getElementById('tab-' + id).classList.toggle('active', id === t);
  });
  var img = document.getElementById('live-img');
  if (img) img.style.visibility = (t === 'record') ? 'visible' : 'hidden';
}
var mr = null, chunks = [], blob = null, previewURL = null;
function setStatus(m, c) {
  var el = document.getElementById('status');
  el.textContent = m;
  el.style.color = c || '#aaa';
}
async function startRec() {
  if (mr && mr.state === 'recording') return;
  var name = document.getElementById('rn').value.trim();
  if (!name) { alert('Enter name first'); return; }
  chunks = []; blob = null;
  setStatus('Requesting camera...', '#aaa');
  var stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
  } catch(e) {
    setStatus('Camera access denied.', '#e55');
    return;
  }
  document.getElementById('rec-dot').style.display = 'block';
  document.getElementById('rec-wrap').style.display = 'none';
  document.getElementById('btn-start').style.display = 'none';
  document.getElementById('btn-stop').style.display = '';
  var mime = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4']
              .find(function(t) { return MediaRecorder.isTypeSupported(t); }) || 'video/webm';
  mr = new MediaRecorder(stream, {mimeType: mime});
  mr.ondataavailable = function(e) { if (e.data.size > 0) chunks.push(e.data); };
  mr.onstop = function() {
    stream.getTracks().forEach(function(t) { t.stop(); });
    document.getElementById('rec-dot').style.display = 'none';
    blob = new Blob(chunks, {type: mime});
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = URL.createObjectURL(blob);
    document.getElementById('preview').src = previewURL;
    document.getElementById('rec-wrap').style.display = 'block';
    setStatus('Recorded ' + (blob.size/1024).toFixed(0) + ' KB', '#7dea7d');
  };
  mr.start(200);
  setStatus('Recording... move face slowly left/right/up/down', '#f0a500');
}
function stopRec() {
  try { if (mr && mr.state !== 'inactive') mr.stop(); } catch(e) { setStatus('Stop error: ' + e.message, '#e55'); }
  document.getElementById('btn-stop').style.display = 'none';
  document.getElementById('btn-start').style.display = '';
}
function resetRec() {
  if (previewURL) { URL.revokeObjectURL(previewURL); previewURL = null; }
  document.getElementById('rec-wrap').style.display = 'none';
  blob = null;
  setStatus('');
}
async function submitRec() {
  if (!blob) { alert('No recording. Record first.'); return; }
  var name = document.getElementById('rn').value.trim();
  if (!name) { alert('Name is required.'); return; }
  setStatus('Uploading and encoding...', '#5b8dee');
  var fd = new FormData();
  fd.append('name',       name);
  fd.append('role',       document.getElementById('rr').value.trim());
  fd.append('contact',    document.getElementById('rc').value.trim());
  fd.append('department', document.getElementById('rd').value.trim());
  fd.append('video',      blob, 'enrollment.mp4');
  try {
    var res  = await fetch('__SAVE_URL__', {method: 'POST', body: fd});
    var data = await res.json();
    if (data.ok) {
      setStatus('Enrolled! Redirecting...', '#7dea7d');
      setTimeout(function() { window.location = '__PERSONS_URL__'; }, 1200);
    } else {
      setStatus('Error: ' + data.error, '#e55');
    }
  } catch(e) {
    setStatus('Upload failed: ' + e.message, '#e55');
  }
}
</script>
</body>
</html>"""


PERSONS_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR.mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Detection constants
FRAMES_TO_CONFIRM  = 6
FRAMES_TO_EXIT     = 45
ALERT_COOLDOWN     = 120
PROCESS_EVERY      = 6    # Run YOLO/face-recog every N frames

# ── Shared state ───────────────────────────────────────────────────────────
_state_lock       = threading.Lock()
_annotated_frames = {}
_det_status       = {}


# ── Config helpers ─────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ── Encoding helpers ───────────────────────────────────────────────────────

_encoding_lock   = threading.Lock()
_encoding_status = {'running': False, 'last': '', 'error': ''}


def _run_encoder_background(reason: str = '') -> None:
    """
    Run video_face_encoder.py in a background thread.

    FIX (v15): After the encoder completes successfully, immediately invalidate
    the recognizer singleton by setting _recognizer = None (not just resetting
    the mtime).  This forces get_recognizer() to do a full reload on the very
    next detection frame, ensuring:
      - Newly enrolled persons are recognised immediately.
      - Deleted persons are de-authorised immediately (their encodings are gone).
    """
    def _task():
        global _recognizer, _recognizer_mtime
        with _encoding_lock:
            _encoding_status['running'] = True
            _encoding_status['error']   = ''
            _encoding_status['last']    = reason
            log.info(f'[ENCODER] Starting ({reason})')
            try:
                result = subprocess.run(
                    [sys.executable, 'video_face_encoder.py'],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    err = (result.stderr or result.stdout or 'unknown').strip()[-300:]
                    _encoding_status['error'] = err
                    log.warning(f'[ENCODER] Failed: {err}')
                else:
                    _encoding_status['error'] = ''
                    if result.stdout:
                        for line in result.stdout.strip().splitlines():
                            log.info(f'[ENCODER] {line}')
                    log.info('[ENCODER] Completed successfully')

                    # FIX: Force recognizer reload — two paths:
                    # 1. Inline get_recognizer() path (used by YOLO detect_and_classify)
                    with _recognizer_lock:
                        _recognizer       = None
                        _recognizer_mtime = 0

                    # 2. Subprocess path — send RELOAD sentinel (0xFFFFFFFF) to
                    #    recog_worker so its in-process FaceRecognizer also reloads.
                    #    Without this, the subprocess NEVER sees the new encodings
                    #    (it loaded them once at startup and has its own memory).
                    _send_reload_to_recog_proc()

            except subprocess.TimeoutExpired:
                _encoding_status['error'] = 'Encoder timed out after 300s'
                log.warning('[ENCODER] Timed out')
            except Exception as e:
                _encoding_status['error'] = str(e)
                log.warning(f'[ENCODER] Exception: {e}')
            finally:
                _encoding_status['running'] = False

    t = threading.Thread(target=_task, daemon=True, name='encoder')
    t.start()


# ── Auth helpers ───────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def _check_credentials(email: str, password: str) -> bool:
    cfg        = load_config()
    portal_cfg = cfg.get('portal', {})
    return (email.strip().lower() == portal_cfg.get('admin_email', '').strip().lower()
            and password == portal_cfg.get('admin_password', ''))


# ── Auth routes ────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)

    if request.method == 'POST':
        form_token    = request.form.get('csrf_token', '')
        session_token = session.get('_csrf_token', '')
        if not secrets.compare_digest(form_token, session_token):
            flash('Invalid request — please try again.', 'error')
            return redirect(url_for('login'))

        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if _check_credentials(email, password):
            session.clear()
            session['logged_in']   = True
            session['user_email']  = email
            session['_csrf_token'] = secrets.token_hex(32)
            if 'remember' in request.form:
                session.permanent = True
            flash('Welcome back!', 'success')
            log.info(f'[AUTH] Login: {email}')
            return redirect(url_for('dashboard'))
        else:
            log.warning(f'[AUTH] Failed login: {email}')
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html', csrf_token=session.get('_csrf_token', ''))


@app.route('/logout')
def logout():
    email = session.get('user_email', 'unknown')
    session.clear()
    log.info(f'[AUTH] Logout: {email}')
    flash('You have been signed out.', 'info')
    return redirect(url_for('login'))


# ── Person helpers ─────────────────────────────────────────────────────────

def list_persons() -> list:
    persons = []
    if not PERSONS_DIR.exists():
        return persons
    for d in sorted(PERSONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / 'meta.txt'
        meta = {
            'id':         d.name,
            'name':       d.name.replace('_', ' ').title(),
            'role':       '', 'contact': '', 'department': '',
            'enrolled':   False, 'added': '',
        }
        if meta_file.exists():
            for line in meta_file.read_text().splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
        for ext in ['*.mp4', '*.mov', '*.avi', '*.webm', '*.mkv']:
            if list(d.glob(ext)):
                meta['enrolled'] = True
                break
        persons.append(meta)
    return persons


def get_stats() -> dict:
    try:
        stats = db.get_stats()
    except Exception as e:
        log.warning(f'get_stats error: {e}')
        stats = {
            'total_events': 0, 'today_authorized': 0,
            'today_unauthorized': 0, 'total_persons': 0,
        }
    stats['total_snapshots'] = len(list(SNAPSHOTS_DIR.glob('*.jpg')))
    stats['unread_alerts']   = alerts_store.unread_count()
    return stats


# ── DB wrappers ────────────────────────────────────────────────────────────

def db_record_entry(person_name, person_id, status, snapshot,
                    camera_id, camera_label='webcam', location=None):
    try:
        return db.record_entry(
            person_name=person_name, person_id=person_id,
            status=status, snapshot=snapshot,
            camera_id=camera_id, camera_label=camera_label,
            location=location,
        )
    except Exception as e:
        log.warning(f'db_record_entry error: {e}')
        return None


def db_record_exit(row_id):
    try:
        db.record_exit(row_id)
    except Exception as e:
        log.warning(f'db_record_exit error: {e}')


# ── Snapshot ───────────────────────────────────────────────────────────────

def save_snapshot(frame, label='UNKNOWN', camera_id=None, event_id=None) -> str:
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = 'intruder' if label == 'UNKNOWN' else 'auth'
    name   = f'{prefix}_{label}_{ts}.jpg'
    path   = SNAPSHOTS_DIR / name
    cv2.imwrite(str(path), frame)
    try:
        db.record_snapshot(filename=name, filepath=str(path),
                           person_name=label, camera_id=camera_id,
                           event_id=event_id)
    except Exception as e:
        log.warning(f'record_snapshot error: {e}')
    return str(path)


# ── OpenCV thread safety ───────────────────────────────────────────────────
# FIX: Replaced single global _cv2_read_lock with a per-camera-index lock
# factory. Camera 0 and Camera 1 no longer block each other.
_cv2_lock_map: dict = {}
_cv2_lock_map_lock  = threading.Lock()


def _get_cam_lock(key) -> threading.Lock:
    """Return (or create) the per-camera read lock for this source key."""
    with _cv2_lock_map_lock:
        if key not in _cv2_lock_map:
            _cv2_lock_map[key] = threading.Lock()
        return _cv2_lock_map[key]


# Keep a global alias for code that still uses _cv2_read_lock directly
_cv2_read_lock = threading.Lock()


def _safe_read(cap, timeout: float = 5.0, cam_key=None):
    """Read a frame with a hard timeout. Uses per-camera lock if cam_key given."""
    result = [False, None]
    lock   = _get_cam_lock(cam_key) if cam_key is not None else _cv2_read_lock

    def _do_read():
        try:
            with lock:
                result[0], result[1] = cap.read()
        except Exception:
            pass

    t = threading.Thread(target=_do_read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        log.warning('[CAMERA] cap.read() timed out')
        return False, None
    return result[0], result[1]


# ── RTSP Frame Buffer ──────────────────────────────────────────────────────

class RtspFrameBuffer:
    """
    Dedicated background-thread RTSP reader.
    Solves 15-20 second RTSP lag by continuously calling cap.grab() to drain
    FFmpeg's internal buffer — so get_frame() always returns the LATEST frame.
    """

    _RTSP_OPTS = (
        'rtsp_transport;tcp'
        '|buffer_size;65536'
        '|max_delay;200000'
        '|reorder_queue_size;0'
        '|analyzeduration;0'
        '|probesize;32'
        '|fflags;nobuffer+discardcorrupt'
        '|flags;low_delay'
    )

    def __init__(self, url: str, reconnect_attempts: int = 5):
        self._url        = url
        self._max_retry  = reconnect_attempts
        self._cap        = None
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._cap_lock   = threading.Lock()
        self._running    = False
        self._connected  = False
        self._fail_streak = 0
        self._MAX_FAILS  = 60

    def _open(self) -> bool:
        env_key = 'OPENCV_FFMPEG_CAPTURE_OPTIONS'
        old_val = os.environ.get(env_key, '')
        os.environ[env_key] = self._RTSP_OPTS
        try:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        finally:
            os.environ[env_key] = old_val

        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10):
            cap.grab()

        ok = cap.grab()
        if not ok:
            cap.release()
            return False

        ret, frm = cap.retrieve()
        if not ret or frm is None:
            cap.release()
            return False

        self._connected = False
        with self._cap_lock:
            old_cap    = self._cap
            self._cap  = cap
        if old_cap:
            try:
                old_cap.release()
            except Exception:
                pass
        with self._frame_lock:
            self._frame = frm
        self._connected   = True
        self._fail_streak = 0
        log.info(f'[RTSP] Connected: {self._url}')
        return True

    def start(self) -> bool:
        for attempt in range(self._max_retry):
            log.info(f'[RTSP] Connect attempt {attempt+1}/{self._max_retry}...')
            if self._open():
                break
            time.sleep(3)
        else:
            log.error(f'[RTSP] All attempts failed: {self._url}')
            return False

        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name='rtsp-grab')
        t.start()
        return True

    def _loop(self):
        while self._running:
            if not self._connected:
                time.sleep(0.05)
                continue

            with self._cap_lock:
                cap = self._cap
                if cap is None:
                    time.sleep(0.05)
                    continue
                ok = cap.grab()
                if ok:
                    ret, frm = cap.retrieve()
                else:
                    ret, frm = False, None

            if ok and ret and frm is not None:
                with self._frame_lock:
                    self._frame = frm
                self._fail_streak = 0
            else:
                self._fail_streak += 1

            if self._fail_streak >= self._MAX_FAILS:
                log.warning('[RTSP] Too many grab failures — reconnecting...')
                self._connected = False
                for attempt in range(self._max_retry):
                    log.info(f'[RTSP] Reconnect attempt {attempt+1}...')
                    if self._open():
                        log.info('[RTSP] Reconnected.')
                        break
                    time.sleep(3)
                else:
                    log.error('[RTSP] Reconnect failed — stopping.')
                    self._running = False

    def get_frame(self):
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_alive(self) -> bool:
        return self._running and self._connected

    def stop(self):
        self._running   = False
        self._connected = False
        time.sleep(0.15)
        with self._cap_lock:
            cap       = self._cap
            self._cap = None
        if cap:
            try:
                cap.release()
            except Exception:
                pass


# ── Camera open helper ─────────────────────────────────────────────────────

def open_camera(src, cam_type: str = 'webcam'):
    src_str = str(src)

    if src_str.startswith('http://') or src_str.startswith('https://'):
        log.info(f'[CAMERA] HTTP stream: {src_str}')
        if not any(x in src_str for x in ['/video', '/shot', '/mjpeg', '.mjpg']):
            src_str = src_str.rstrip('/') + '/video'
        cap = cv2.VideoCapture(src_str)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frm = _safe_read(cap, timeout=10.0)
        if ret and frm is not None:
            return cap
        cap.release()
        return None

    if cam_type == 'rtsp':
        from urllib.parse import urlparse, unquote, quote as _quote
        log.info(f'[CAMERA] RTSP: {src_str}')
        try:
            parsed   = urlparse(src_str)
            username = unquote(parsed.username or '')
            password = unquote(parsed.password or '')
            host     = parsed.hostname or ''
            port     = parsed.port or 554
            path     = parsed.path or '/'
        except Exception:
            username = password = ''
            host = port = path = ''

        base_paths = [
            path,
            path.replace('/101', '/102').replace('/1', '/2') if '/1' in path else '/Streaming/Channels/102',
            '/h264/ch1/main/av_stream',
        ]
        candidates = [(f'rtsp://{host}:{port}{bp}', 'TCP') for bp in base_paths]
        candidates.append((f'rtsp://{host}:{port}{path}', 'UDP'))

        for open_url, transport in candidates:
            opts = f'rtsp_transport;{transport.lower()}|allowed_media_types;video|timeout;10000000'
            old_opts = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', '')
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = opts

            if username and password:
                safe_pass = _quote(password, safe='')
                final_url = (f'rtsp://{_quote(username, safe="")}:{safe_pass}'
                             f'@{host}:{port}{open_url[open_url.index(str(port))+len(str(port)):]}')
            else:
                final_url = open_url

            cap = cv2.VideoCapture(final_url, cv2.CAP_FFMPEG)
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = old_opts

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frm = _safe_read(cap, timeout=12.0)
            if ret and frm is not None:
                log.info(f'[CAMERA] RTSP OK ({transport}): {open_url}')
                return cap
            cap.release()

        log.warning('[CAMERA] RTSP: all attempts failed')
        return None

    # Webcam
    idx = int(src) if not isinstance(src, int) else src
    cam_lock = _get_cam_lock(idx)
    backends = (
        [(cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_ANY, 'ANY')]
        if platform.system() == 'Windows'
        else [(cv2.CAP_ANY, 'ANY')]
    )
    for backend, bname in backends:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(8):
            with cam_lock:
                ret, frm = cap.read()
            if ret and frm is not None:
                log.info(f'[CAMERA] Webcam {idx} via {bname}')
                return cap
            time.sleep(0.1)
        cap.release()
    log.error(f'[CAMERA] Cannot open webcam {idx}')
    return None


# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_boxes(frame, results):
    for r in results:
        top, right, bottom, left = r['location']
        color = (0, 210, 0) if r['authorized'] else (0, 0, 220)
        label = r['name']
        if r['authorized']:
            label += f" {r['confidence']*100:.0f}%"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (left, bottom), (left+tw+8, bottom+th+10), color, cv2.FILLED)
        cv2.putText(frame, label, (left+4, bottom+th+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def draw_status_bar(frame, intruder_active: bool, cam_label: str,
                    face_count: int = 0, yolo_persons: int = 0,
                    auth_count: int = 0, unauth_count: int = 0):
    h, w = frame.shape[:2]
    bar_color = (0, 0, 180) if intruder_active else (0, 130, 0)
    cv2.rectangle(frame, (0, 0), (w, 36), bar_color, cv2.FILLED)
    status_txt = '!! INTRUSION DETECTED' if intruder_active else 'OK  ALL CLEAR'
    cv2.putText(frame, status_txt, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if yolo_persons > 0:
        info_txt = f'Persons:{yolo_persons}  Auth:{auth_count}  Unk:{unauth_count}'
    else:
        info_txt = f'Faces:{face_count}'
    (tw, _), _ = cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, info_txt, (w - tw - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    cv2.rectangle(frame, (0, h-28), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, ts,        (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, cam_label, (w//2-60, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (91, 141, 238), 1)
    return frame


# ── Camera Blockage Detector ───────────────────────────────────────────────

class BlockageDetector:
    def __init__(self,
                 brightness_thresh:   float = 15.0,
                 variance_thresh:     float = 20.0,
                 edge_density_thresh: float = 0.002,
                 confirm_seconds:     float = 3.0,
                 cooldown_seconds:    float = 120.0):
        self.brightness_thresh   = brightness_thresh
        self.variance_thresh     = variance_thresh
        self.edge_density_thresh = edge_density_thresh
        self.confirm_seconds     = confirm_seconds
        self.cooldown_seconds    = cooldown_seconds
        self._suspect_since = None
        self._last_alert_at = 0.0
        self._blocked       = False

    def analyse(self, frame) -> dict:
        result = dict(blocked=False, just_triggered=False, just_cleared=False,
                      reason='', brightness=0.0, variance=0.0, edge_density=0.0)
        if frame is None:
            return result

        grey         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness   = float(np.mean(grey))
        variance     = float(np.var(grey))
        edges        = cv2.Canny(grey, 30, 100)
        edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

        result.update(brightness=round(brightness, 2),
                      variance=round(variance, 2),
                      edge_density=round(edge_density, 5))

        dark      = brightness   < self.brightness_thresh
        uniform   = variance     < self.variance_thresh
        no_edges  = edge_density < self.edge_density_thresh
        suspect   = dark and uniform and no_edges
        now       = time.time()

        reasons = []
        if dark:     reasons.append(f'blackout(b={brightness:.1f})')
        if uniform:  reasons.append(f'uniform(v={variance:.1f})')
        if no_edges: reasons.append(f'no-edges(d={edge_density:.4f})')

        if suspect:
            result['reason'] = '; '.join(reasons)
            if self._suspect_since is None:
                self._suspect_since = now
            elapsed = now - self._suspect_since
            if elapsed >= self.confirm_seconds and not self._blocked:
                if (now - self._last_alert_at) >= self.cooldown_seconds:
                    self._blocked            = True
                    result['blocked']        = True
                    result['just_triggered'] = True
                    self._last_alert_at      = now
                else:
                    self._blocked     = True
                    result['blocked'] = True
            else:
                result['blocked'] = self._blocked
        else:
            if self._blocked:
                result['just_cleared'] = True
            self._suspect_since = None
            self._blocked       = False
            result['blocked']   = False

        return result

    def reset(self):
        self._suspect_since = None
        self._last_alert_at = 0.0
        self._blocked       = False


def draw_blocked_overlay(frame, reason: str = '') -> np.ndarray:
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    main_txt = '!! CAMERA BLOCKED / BLACKOUT'
    scale    = max(0.5, w / 800)
    (tw, th), _ = cv2.getTextSize(main_txt, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cx = (w - tw) // 2
    cy = h // 2 - 20
    cv2.putText(frame, main_txt, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    if reason:
        sub_scale = max(0.35, w / 1200)
        (sw, _), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
        cv2.putText(frame, reason, ((w - sw) // 2, cy + th + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 200, 200), 1, cv2.LINE_AA)

    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    cv2.rectangle(frame, (0, h-28), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, ts, (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return frame


# ── Recognizer loader ──────────────────────────────────────────────────────

_recognizer       = None
_recognizer_lock  = threading.Lock()
_recognizer_mtime = 0


def get_recognizer():
    """
    Return the shared FaceRecognizer, reloading from disk if pkl changed.

    FIX (v15): Setting _recognizer = None (in _run_encoder_background) is the
    primary invalidation signal. The mtime check is a secondary fallback for
    external changes to face_encodings.pkl.

    FIX (v16): If pkl exists but the loaded recognizer has 0 encodings,
    force a reload — handles the case where pkl was written after the
    recognizer was first constructed (startup race).
    """
    global _recognizer, _recognizer_mtime
    pkl = 'face_encodings.pkl'
    try:
        mtime = os.path.getmtime(pkl) if os.path.exists(pkl) else 0
    except Exception:
        mtime = 0

    with _recognizer_lock:
        needs_reload = (
            _recognizer is None
            or mtime != _recognizer_mtime
            or (mtime > 0 and _recognizer is not None
                and len(_recognizer._known_encodings) == 0)
        )
        if needs_reload:
            try:
                from face_recognizer import FaceRecognizer
                _recognizer       = FaceRecognizer()
                _recognizer_mtime = mtime
                log.info(f'[RECOGNIZER] Loaded/reloaded '
                         f'({len(_recognizer._known_encodings)} encodings).')
            except Exception as e:
                log.warning(f'[RECOGNIZER] Load failed: {e}')
        return _recognizer


# ── Per-camera detection thread ────────────────────────────────────────────

_started_cams = set()
_start_lock   = threading.Lock()

import queue as _queue
import pickle as _pickle
import struct as _struct

# ── Global shared recog subprocess ────────────────────────────────────────
# Single subprocess shared by all cameras — prevents multi-dlib crash.
# _GLOBAL_RECOG_Q_IMPL was created before gevent patching (see top of file).

_GLOBAL_RECOG_PROC        = None
_GLOBAL_RECOG_PROC_LOCK   = threading.Lock()
_GLOBAL_RECOG_Q           = _GLOBAL_RECOG_Q_IMPL   # alias — pre-patching queue
_GLOBAL_RECOG_RESULTS     = {}
_GLOBAL_RECOG_RESULTS_LOCK = threading.Lock()


def _send_reload_to_recog_proc() -> None:
    """
    Send the RELOAD sentinel (4 bytes 0xFFFFFFFF) to the global recog subprocess.

    recog_worker.py v3 reads this sentinel and calls FaceRecognizer() fresh,
    reloading face_encodings.pkl from disk.  Without this, the subprocess keeps
    its old encodings forever — setting _recognizer=None in portal only affects
    the inline path, NOT the subprocess which has its own memory.
    """
    try:
        with _GLOBAL_RECOG_PROC_LOCK:
            proc = _GLOBAL_RECOG_PROC
        if proc and proc.poll() is None and proc.stdin:
            proc.stdin.write(_struct.pack('<I', 0xFFFFFFFF))
            proc.stdin.flush()
            log.info('[RECOG] Reload sentinel sent to recog_worker subprocess')
    except Exception as e:
        log.warning(f'[RECOG] Could not send reload sentinel: {e}')

# ── Model init lock — serialise native DLL loading ────────────────────────
_MODEL_INIT_LOCK = threading.Lock()

# ── YOLO inference locks — one per model type ─────────────────────────────
_YOLO_PERSON_LOCK = threading.Lock()
_YOLO_HAZARD_LOCK = threading.Lock()
_YOLO_INFER_LOCK  = _YOLO_PERSON_LOCK   # backwards-compat alias


def _start_global_recog_proc():
    try:
        log_fh = open('logs/recog_worker.log', 'a')
        p = subprocess.Popen(
            [sys.executable, 'recog_worker.py'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log_fh,
        )
        log.info(f'[RECOG] Global recog subprocess started PID={p.pid}')
        return p
    except Exception as e:
        log.error(f'[RECOG] Failed to start subprocess: {e}')
        return None


def _global_recog_dispatcher():
    """
    Single background thread that owns the recog subprocess.
    Uses the pre-patching stdlib queue (_GLOBAL_RECOG_Q_IMPL) so it never
    blocks gevent's event loop even if monkey_patch ran.
    """
    global _GLOBAL_RECOG_PROC
    with _GLOBAL_RECOG_PROC_LOCK:
        _GLOBAL_RECOG_PROC = _start_global_recog_proc()

    while True:
        # FIX: use timeout=1.0 so the thread doesn't block forever if the
        # queue is empty under gevent (avoids LoopExit in some edge cases).
        try:
            item = _GLOBAL_RECOG_Q.get(timeout=1.0)
        except _queue.Empty:
            continue

        if item is None:
            break
        cam_id, frame = item

        with _GLOBAL_RECOG_PROC_LOCK:
            proc = _GLOBAL_RECOG_PROC

        if proc is None or proc.poll() is not None:
            log.warning('[RECOG] Subprocess died — restarting')
            with _GLOBAL_RECOG_PROC_LOCK:
                _GLOBAL_RECOG_PROC = _start_global_recog_proc()
                proc = _GLOBAL_RECOG_PROC
            if proc is None:
                with _GLOBAL_RECOG_RESULTS_LOCK:
                    _GLOBAL_RECOG_RESULTS[cam_id] = []
                continue

        try:
            raw = _pickle.dumps(frame, protocol=4)
            hdr = _struct.pack('<I', len(raw))
            proc.stdin.write(hdr + raw)
            proc.stdin.flush()
            hdr2 = proc.stdout.read(4)
            if len(hdr2) < 4:
                raise IOError('short header from recog worker')
            length = _struct.unpack('<I', hdr2)[0]
            data   = proc.stdout.read(length)
            res    = _pickle.loads(data) if len(data) == length else []
        except Exception as e:
            log.warning(f'[RECOG] IPC error for {cam_id}: {e}')
            try:
                proc.kill()
            except Exception:
                pass
            with _GLOBAL_RECOG_PROC_LOCK:
                _GLOBAL_RECOG_PROC = _start_global_recog_proc()
            res = []

        with _GLOBAL_RECOG_RESULTS_LOCK:
            _GLOBAL_RECOG_RESULTS[cam_id] = res


# Start dispatcher thread once
_global_recog_thread = threading.Thread(
    target=_global_recog_dispatcher, daemon=True, name='recog-global')
_global_recog_thread.start()


def ensure_detection_thread(cam_cfg: dict) -> None:
    """
    Start a per-camera detection worker thread (idempotent).

    FIX (v15):
      - Outer retry loop: if the worker crashes it sleeps 5 s and restarts.
      - Per-camera webcam lock via _get_cam_lock(cam_src) instead of global.
      - Adaptive sleep: always sleep at least 33 ms between frames (30 fps cap).
      - Full-frame fallback skipped when YOLO has resolved all persons.
      - JPEG encode quality lowered to 65 (was 75) for faster encode.
    """
    cam_id = cam_cfg.get('id', 'cam0')
    with _start_lock:
        if cam_id in _started_cams:
            return
        _started_cams.add(cam_id)

    def _worker():
        label    = cam_cfg.get('label', cam_id)
        source   = cam_cfg.get('source', 'webcam')
        rtsp_url = cam_cfg.get('rtsp_url', '')
        location = cam_cfg.get('location_name', '')

        _url = rtsp_url or ''
        if source == 'rtsp' or _url.startswith('rtsp://'):
            cam_type = 'rtsp'
            cam_src  = _url
        elif _url.startswith('http://') or _url.startswith('https://'):
            cam_type = 'http'
            cam_src  = _url
        else:
            cam_type = 'webcam'
            cam_src  = int(source) if source not in ('webcam', 'rtsp') else 0

        log.info(f'[DET] Thread starting for {label} ({cam_id})')

        # ── Camera open ────────────────────────────────────────────────
        rtsp_buf = None
        cap      = None

        if cam_type == 'rtsp':
            rtsp_buf = RtspFrameBuffer(
                url=cam_src,
                reconnect_attempts=cam_cfg.get('reconnect_attempts', 5),
            )
            if not rtsp_buf.start():
                log.error(f'[DET] Cannot open RTSP {label}')
                with _state_lock:
                    _det_status[cam_id] = {'intruder': False, 'faces': 0, 'running': False}
                with _start_lock:
                    _started_cams.discard(cam_id)
                return
        else:
            for attempt in range(cam_cfg.get('reconnect_attempts', 5)):
                cap = open_camera(cam_src, cam_type)
                if cap:
                    break
                log.warning(f'[DET] Attempt {attempt+1} failed for {label}')
                time.sleep(3)
            if not cap:
                log.error(f'[DET] Cannot open {label}')
                with _state_lock:
                    _det_status[cam_id] = {'intruder': False, 'faces': 0, 'running': False}
                with _start_lock:
                    _started_cams.discard(cam_id)
                return

        # Per-camera webcam lock
        cam_lock = _get_cam_lock(cam_src if cam_type == 'webcam' else cam_id)

        try:
            db.upsert_camera(
                cam_id=cam_id, label=label, source=source, rtsp_url=rtsp_url,
                location_name=location,
                floor=cam_cfg.get('floor', ''),
                building=cam_cfg.get('building', ''),
                location_lat=cam_cfg.get('location_lat'),
                location_lng=cam_cfg.get('location_lng'),
            )
        except Exception as e:
            log.warning(f'[DET] upsert_camera error: {e}')

        # ── Load config and models ─────────────────────────────────────
        _full_cfg   = load_config()
        _ycfg       = _full_cfg.get('yolo', {})
        _process_every = _ycfg.get('process_every', PROCESS_EVERY)
        _bcfg          = _full_cfg.get('blockage_detection', {})
        blockage_alert_on_clear = _bcfg.get('alert_on_clear', True)

        log.info(f'[DET] {label}: acquiring model-init lock...')
        with _MODEL_INIT_LOCK:
            log.info(f'[DET] {label}: loading models...')
            _yolo            = get_yolo_detector(_ycfg)
            _hcfg            = _full_cfg.get('hazard', {})
            _hazard          = get_hazard_detector(_hcfg) if _hcfg.get('enabled') else None
            _recognizer_inst = get_recognizer()
        log.info(f'[DET] {label}: models ready — starting frame loop')

        face_sessions      = {}
        face_counters      = {}
        absent_counters    = {}
        last_alert_time    = {}
        frame_count        = 0
        consecutive_fails  = 0
        MAX_FAILS          = 30
        last_yolo_dets     = []   # FIX: persist last YOLO detections across frames
        last_yolo_frame    = 0    # frame_count when last_yolo_dets was updated

        blockage_detector = BlockageDetector(
            brightness_thresh   = _bcfg.get('brightness_thresh',   15.0),
            variance_thresh     = _bcfg.get('variance_thresh',     20.0),
            edge_density_thresh = _bcfg.get('edge_density_thresh', 0.002),
            confirm_seconds     = _bcfg.get('confirm_seconds',     3.0),
            cooldown_seconds    = _bcfg.get('cooldown_seconds',    120.0),
        )

        try:
            while True:
                loop_start = time.time()

                # ── Grab frame ────────────────────────────────────────
                if cam_type == 'rtsp':
                    frame = rtsp_buf.get_frame()
                    ret   = frame is not None
                    if not ret and not rtsp_buf.is_alive:
                        log.error(f'[DET] {label}: RTSP buffer died — stopping')
                        break
                else:
                    with cam_lock:
                        ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= MAX_FAILS and cam_type != 'rtsp':
                        log.warning(f'[DET] {label}: too many failures — reconnecting')
                        cap.release()
                        cap = None
                        for attempt in range(cam_cfg.get('reconnect_attempts', 5)):
                            cap = open_camera(cam_src, cam_type)
                            if cap:
                                consecutive_fails = 0
                                blockage_detector.reset()
                                break
                            time.sleep(3)
                        if not cap:
                            log.error(f'[DET] {label}: reconnect failed — stopping')
                            break
                    # FIX: always sleep — prevents 100% CPU spin on webcam
                    time.sleep(0.033)
                    continue

                consecutive_fails = 0
                frame_count      += 1

                if frame.size == 0 or len(frame.shape) != 3:
                    time.sleep(0.033)
                    continue

                # ── Blockage detection ─────────────────────────────────
                blk = blockage_detector.analyse(frame)

                if blk['just_triggered']:
                    reason_str = blk.get('reason', '')
                    log.warning(f'[BLOCKAGE] {label}: BLOCKED — {reason_str}')
                    snap_path = None
                    try:
                        ts_str    = datetime.now().strftime('%Y%m%d_%H%M%S')
                        snap_name = f'blocked_{cam_id}_{ts_str}.jpg'
                        snap_path = str(SNAPSHOTS_DIR / snap_name)
                        cv2.imwrite(snap_path, frame)
                    except Exception as se:
                        log.warning(f'[BLOCKAGE] snapshot error: {se}')
                    try:
                        alerts_store.AlertSystem().send_camera_blocked_alert(
                            camera=label, reason=reason_str, snapshot_path=snap_path)
                    except Exception as ae:
                        log.warning(f'[BLOCKAGE] alert error: {ae}')
                    with _state_lock:
                        _det_status[cam_id] = {'intruder': False, 'faces': 0,
                                               'running': True, 'blocked': True}

                elif blk['just_cleared']:
                    log.info(f'[BLOCKAGE] {label}: camera CLEAR again.')
                    if blockage_alert_on_clear:
                        try:
                            alerts_store.AlertSystem().send_camera_cleared_alert(camera=label)
                        except Exception as ae:
                            log.warning(f'[BLOCKAGE] cleared alert error: {ae}')

                if blk['blocked']:
                    display = draw_blocked_overlay(frame.copy(), blk.get('reason', ''))
                    try:
                        ok, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if ok and buf is not None:
                            with _state_lock:
                                _annotated_frames[cam_id] = buf.tobytes()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    continue

                display = frame.copy()

                # ── Resize to 480p for HAZARD detection only ───────────
                # YOLO person detection: pass the ORIGINAL frame — yolo_detector.py
                # v4 handles internal resizing to 640×640 for OpenVINO and maps
                # all boxes back to original frame coordinates automatically.
                # Hazard model still uses 480p to save CPU.
                _h, _w = frame.shape[:2]
                _hazard_scale = min(1.0, 480 / _h)
                if _hazard_scale < 1.0:
                    hazard_frame = cv2.resize(
                        frame, (int(_w * _hazard_scale), 480),
                        interpolation=cv2.INTER_LINEAR)
                else:
                    hazard_frame = frame

                # ── Hazard detection ───────────────────────────────────
                _cam_hazard_on = (cam_type == 'rtsp') and cam_cfg.get('hazard_enabled', True)
                if (_hazard and _cam_hazard_on
                        and frame_count % _hcfg.get('process_every', 15) == 0):
                    try:
                        hazard_results = _hazard.detect(
                            hazard_frame, infer_lock=_YOLO_HAZARD_LOCK)
                        if hazard_results:
                            display = _hazard.draw(display, hazard_results)
                            for hz in hazard_results:
                                _hazard.maybe_alert(hz, label, alerts_store, cam_id)
                    except Exception as hze:
                        log.warning(f'[HAZARD] detect error: {hze}')

                # ── Integrated YOLO + face-auth ────────────────────────
                # Pass the ORIGINAL frame — yolo_detector v4 handles 640 resize
                # internally for OpenVINO and returns boxes in original coords.
                # NO box rescaling needed here anymore.
                yolo_persons = 0
                if _yolo and frame_count % _process_every == 0:
                    try:
                        _recognizer_inst = get_recognizer()
                        new_dets = _yolo.detect_and_classify(
                            frame, _recognizer_inst,
                            infer_lock=_YOLO_PERSON_LOCK)

                        # FIX: persist last detections so boxes stay on screen
                        # between inference frames (every Nth frame only)
                        last_yolo_dets  = new_dets
                        last_yolo_frame = frame_count

                    except Exception as ye:
                        log.warning(f'[YOLO] detect_and_classify error: {ye}')

                # Clear stale detections after 2× process_every frames (person left)
                if frame_count - last_yolo_frame > _process_every * 2:
                    last_yolo_dets = []

                yolo_dets    = last_yolo_dets
                yolo_persons = sum(1 for d in yolo_dets if d.get('class_id', -1) == 0)

                # Draw boxes every frame using the persisted detections
                if yolo_dets:
                    display = _yolo.draw(display, yolo_dets)

                # ── Full-frame fallback (RTSP only, skip if YOLO resolved all) ──
                # FIX: Skip IPC call entirely when YOLO already found and
                # resolved all visible persons as authorised.
                _all_yolo_resolved = (yolo_persons > 0 and all(
                    d.get('authorized') is not None for d in yolo_dets))

                _FULLFRAME_EVERY = 20
                if (cam_type == 'rtsp'
                        and frame_count % _FULLFRAME_EVERY == 0
                        and not _all_yolo_resolved):
                    try:
                        _h_ff, _w_ff = frame.shape[:2]
                        _sc_ff = min(1.0, 480 / _h_ff)
                        _small_ff = (cv2.resize(frame,
                                                (int(_w_ff * _sc_ff), 480),
                                                interpolation=cv2.INTER_LINEAR)
                                     if _sc_ff < 1.0 else frame)
                        _GLOBAL_RECOG_Q.put_nowait((cam_id, _small_ff))
                    except _queue.Full:
                        pass

                # Read latest full-frame result
                with _GLOBAL_RECOG_RESULTS_LOCK:
                    fullframe_results = _GLOBAL_RECOG_RESULTS.get(cam_id) or []

                # ── Merge YOLO + fullframe results ─────────────────────
                yolo_names = {d['name'] for d in yolo_dets
                              if d.get('name') and d['name'] != 'UNKNOWN'}

                extra_results = [r for r in fullframe_results
                                 if r.get('name') and
                                    r['name'] != 'UNKNOWN' and
                                    r['name'] not in yolo_names]
                if extra_results:
                    draw_boxes(display, extra_results)

                combined_results = []
                for d in yolo_dets:
                    if d.get('class_id', -1) != 0:   # only persons
                        continue
                    face_res = d.get('face_result') or {}
                    d_name   = d.get('name')          # None / 'UNKNOWN' / actual name
                    d_auth   = d.get('authorized')    # None / True / False

                    # Upgrade UNKNOWN from YOLO if fullframe found them
                    if (d_name is None or d_name == 'UNKNOWN') and d_auth is not True:
                        for fr in fullframe_results:
                            if fr.get('authorized') and fr.get('name') and \
                                    fr['name'] != 'UNKNOWN':
                                d_name   = fr['name']
                                d_auth   = True
                                face_res = fr
                                break

                    # FIX: Always include detected persons in combined_results.
                    # Previously entries with d_name=None were silently dropped,
                    # meaning unknown persons (no face found in crop) never
                    # triggered alarms or appeared in access logs.
                    # Now: name=None → treat as 'UNKNOWN', authorized=False.
                    if d_name is None:
                        d_name = 'UNKNOWN'
                    if d_auth is None:
                        d_auth = False   # unresolved face → treat as unauthorized

                    combined_results.append({
                        'name':       d_name,
                        'authorized': d_auth,
                        'person_id':  d.get('person_id') or face_res.get('person_id'),
                        'confidence': face_res.get('confidence', 0),
                        'location':   d.get('location'),
                    })
                combined_results.extend(extra_results)

                # ── Access logging ─────────────────────────────────────
                seen_names = set()
                intruder   = False

                for r in combined_results:
                    name = r['name']
                    auth = r['authorized']
                    seen_names.add(name)

                    face_counters[name]   = face_counters.get(name, 0) + 1
                    absent_counters[name] = 0

                    if not auth:
                        intruder = True

                    if face_counters[name] == FRAMES_TO_CONFIRM:
                        snap_label = name if auth else 'UNKNOWN'
                        snap_path  = save_snapshot(frame, label=snap_label,
                                                   camera_id=cam_id)
                        if not auth:
                            now = time.time()
                            if now - last_alert_time.get(name, 0) > ALERT_COOLDOWN:
                                last_alert_time[name] = now
                                try:
                                    alerts_store.AlertSystem().send_intrusion_alert(
                                        snap_path, camera=label)
                                except Exception as ae:
                                    log.warning(f'[ALERT] {ae}')

                        row_id = db_record_entry(
                            person_name=name,
                            person_id=r.get('person_id'),
                            status='AUTHORIZED' if auth else 'UNAUTHORIZED',
                            snapshot=snap_path,
                            camera_id=cam_id,
                            camera_label=label,
                            location=location,
                        )
                        face_sessions[name] = row_id

                for name in list(face_counters.keys()):
                    if name not in seen_names:
                        absent_counters[name] = absent_counters.get(name, 0) + 1
                        if absent_counters[name] >= FRAMES_TO_EXIT:
                            if name in face_sessions:
                                db_record_exit(face_sessions.pop(name))
                            face_counters.pop(name, None)
                            absent_counters.pop(name, None)

                # ── Status bar ─────────────────────────────────────────
                auth_count   = sum(1 for r in combined_results if r.get('authorized') is True)
                unauth_count = sum(1 for r in combined_results if r.get('authorized') is False)
                draw_status_bar(display, intruder, label,
                                face_count=len(combined_results),
                                yolo_persons=yolo_persons,
                                auth_count=auth_count,
                                unauth_count=unauth_count)

                with _state_lock:
                    _det_status[cam_id] = {
                        'intruder':     intruder,
                        'faces':        len(combined_results),
                        'persons_yolo': yolo_persons,
                        'authorized':   auth_count,
                        'unauthorized': unauth_count,
                        'running':      True,
                    }

                # ── JPEG encode ────────────────────────────────────────
                if (display is not None and display.size > 0
                        and len(display.shape) == 3
                        and display.shape[0] > 0 and display.shape[1] > 0):
                    try:
                        ok, buf = cv2.imencode(
                            '.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if ok and buf is not None:
                            with _state_lock:
                                _annotated_frames[cam_id] = buf.tobytes()
                    except Exception as enc_err:
                        log.warning(f'[DET] {label}: imencode failed: {enc_err}')

                # FIX: Adaptive sleep — cap at ~30 fps, never spin-burn CPU.
                elapsed = time.time() - loop_start
                sleep_s = max(0.0, 0.033 - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except Exception as e:
            log.error(f'[DET] Thread crashed for {cam_id}: {e}', exc_info=True)
        finally:
            if cap:
                cap.release()
            if rtsp_buf:
                rtsp_buf.stop()
            with _GLOBAL_RECOG_RESULTS_LOCK:
                _GLOBAL_RECOG_RESULTS.pop(cam_id, None)
            with _start_lock:
                _started_cams.discard(cam_id)
            log.info(f'[DET] Thread exited for {label} ({cam_id})')

    def _worker_with_restart():
        """
        FIX: Outer restart loop — if the worker crashes it waits 5 s and
        restarts automatically rather than dying permanently.
        """
        while True:
            try:
                _worker()
            except Exception as e:
                log.error(f'[DET] Worker outer crash for {cam_id}: {e}', exc_info=True)

            # Clean up started_cams so the restart is allowed
            with _start_lock:
                _started_cams.discard(cam_id)

            log.info(f'[DET] Worker for {cam_id} will restart in 5 s...')
            time.sleep(5)

            # Re-add to started_cams before restarting
            with _start_lock:
                if cam_id in _started_cams:
                    break   # another thread started it already
                _started_cams.add(cam_id)

    t = threading.Thread(target=_worker_with_restart,
                         daemon=True, name=f'det-{cam_id}')
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
@login_required
def dashboard():
    cfg    = load_config()
    stats  = get_stats()
    cams   = cfg.get('cameras', [])
    events = []
    try:
        events = db.get_recent_events(limit=10)
    except Exception as e:
        log.warning(f'dashboard events error: {e}')
    return render_template('dashboard.html', stats=stats, cameras=cams, events=events)


@app.route('/persons')
@login_required
def persons():
    return render_template('persons.html', persons=list_persons())


@app.route('/persons/add')
@login_required
def add_person():
    cfg      = load_config()
    cams     = cfg.get('cameras', [])
    feed_cam = cams[0]['id'] if cams else 'cam0'
    html = (ADD_PERSON_HTML
            .replace('__PERSONS_URL__', url_for('persons'))
            .replace('__ADD_URL__',     url_for('save_person_upload'))
            .replace('__SAVE_URL__',    url_for('save_recorded'))
            .replace('__FEED_URL__',    url_for('video_feed', cam_id=feed_cam)))
    return html


@app.route('/persons/save_upload', methods=['POST'])
@login_required
def save_person_upload():
    name    = request.form.get('name', '').strip()
    role    = request.form.get('role', '').strip()
    contact = request.form.get('contact', '').strip()
    dept    = request.form.get('department', '').strip()

    if not name:
        flash('Name is required.', 'error')
        return redirect(url_for('add_person'))

    person_id = name.lower().replace(' ', '_') + f'_{int(time.time())}'
    folder    = PERSONS_DIR / person_id
    folder.mkdir(parents=True, exist_ok=True)

    video = request.files.get('video')
    if not video or video.filename == '':
        flash('Video file required.', 'error')
        return redirect(url_for('add_person'))

    ext = video.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_VIDEO:
        flash('Invalid video format.', 'error')
        return redirect(url_for('add_person'))

    video.save(str(folder / f'enrollment.{ext}'))
    (folder / 'meta.txt').write_text(
        f'name:{name}\nrole:{role}\ncontact:{contact}\ndepartment:{dept}\n'
        f'added:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    try:
        db.upsert_person(person_id=person_id, name=name, role=role,
                         contact=contact, department=dept)
    except Exception as e:
        log.warning(f'upsert_person error: {e}')

    _run_encoder_background(reason=f'enrolled {name}')
    flash(f'"{name}" enrolled — encoding started. '
          f'Recognition active within ~30 seconds.', 'success')
    return redirect(url_for('persons'))


@app.route('/persons/save_recorded', methods=['POST'])
@login_required
def save_recorded():
    name    = request.form.get('name', '').strip()
    role    = request.form.get('role', '').strip()
    contact = request.form.get('contact', '').strip()
    dept    = request.form.get('department', '').strip()

    if not name:
        return jsonify({'ok': False, 'error': 'Name required'})

    person_id = name.lower().replace(' ', '_') + f'_{int(time.time())}'
    folder    = PERSONS_DIR / person_id
    folder.mkdir(parents=True, exist_ok=True)

    video = request.files.get('video')
    if not video:
        return jsonify({'ok': False, 'error': 'No video'})

    video.save(str(folder / 'enrollment.webm'))
    (folder / 'meta.txt').write_text(
        f'name:{name}\nrole:{role}\ncontact:{contact}\ndepartment:{dept}\n'
        f'added:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    try:
        db.upsert_person(person_id=person_id, name=name, role=role,
                         contact=contact, department=dept)
    except Exception as e:
        log.warning(f'upsert_person error: {e}')

    _run_encoder_background(reason=f'enrolled {name}')
    return jsonify({'ok': True, 'message': 'Enrolled — encoding started'})


@app.route('/persons/delete/<person_id>', methods=['POST'])
@login_required
def delete_person(person_id):
    folder = PERSONS_DIR / secure_filename(person_id)
    if folder.exists():
        shutil.rmtree(str(folder))
    try:
        db.delete_person(person_id)
    except AttributeError:
        try:
            db.deactivate_person(person_id)
        except Exception as e:
            log.warning(f'deactivate_person error: {e}')
    except Exception as e:
        log.warning(f'delete_person error: {e}')

    # FIX: immediately invalidate recognizer so deleted person is de-authorised
    # on the NEXT frame, before the encoder even finishes.
    global _recognizer, _recognizer_mtime
    with _recognizer_lock:
        _recognizer       = None
        _recognizer_mtime = 0

    _run_encoder_background(reason=f'deleted {person_id}')
    flash('Person removed — re-encoding started. '
          'They will be unauthorised immediately.', 'success')
    return redirect(url_for('persons'))


@app.route('/encoder_status')
@login_required
def encoder_status():
    import pickle as _pkl
    enc_info = {'persons': 0, 'encodings': 0, 'error': ''}
    try:
        with open('face_encodings.pkl', 'rb') as _f:
            _d = _pkl.load(_f)
        enc_info['encodings'] = len(_d.get('encodings', _d.get('embeddings', [])))
        enc_info['persons']   = len(set(_d.get('names', [])))
    except Exception as _e:
        enc_info['error'] = str(_e)
    return jsonify({
        'encoder': _encoding_status,
        'pkl':     enc_info,
        'message': (
            f"PKL has {enc_info['encodings']} encodings for {enc_info['persons']} persons. "
            + (f"Encoder error: {enc_info['error']}" if enc_info['error'] else
               ('Encoder running...' if _encoding_status['running'] else 'Encoder idle.'))
        )
    })


@app.route('/logs')
@login_required
def logs():
    PAGE_SIZE = 50
    page      = request.args.get('page', 1, type=int)
    events    = []
    total     = 0
    try:
        all_events = db.get_recent_events(limit=500)
        total      = len(all_events)
        start      = (page - 1) * PAGE_SIZE
        events     = all_events[start: start + PAGE_SIZE]
    except Exception as e:
        log.warning(f'logs page error: {e}')
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    return render_template('logs.html', events=events,
                           total=total, page=page, pages=pages)


@app.route('/snapshots')
@login_required
def snapshots():
    files = sorted(SNAPSHOTS_DIR.glob('*.jpg'),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    items = [{
        'name': f.name,
        'ts':   datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        'size': f'{f.stat().st_size // 1024} KB',
    } for f in files]
    return render_template('snapshots.html', snapshots=items)


@app.route('/snapshots/img/<filename>')
@login_required
def snapshot_img(filename):
    return send_from_directory(str(SNAPSHOTS_DIR), filename)


@app.route('/snapshots/delete/<filename>', methods=['POST'])
@login_required
def delete_snapshot(filename):
    path = SNAPSHOTS_DIR / secure_filename(filename)
    if path.exists():
        path.unlink()
        flash('Snapshot deleted.', 'success')
    return redirect(url_for('snapshots'))


@app.route('/alerts')
@login_required
def alert_list():
    alert_data = alerts_store.get_alerts(limit=50)
    alerts_store.mark_all_read()
    return render_template('alerts.html', alerts=alert_data)


@app.route('/alerts/clear', methods=['POST'])
@login_required
def clear_alerts():
    alerts_store.clear_alerts()
    flash('All alerts cleared.', 'success')
    return redirect(url_for('alert_list'))


@app.route('/cameras')
@login_required
def cameras():
    cfg      = load_config()
    cam_list = cfg.get('cameras', [])
    for cam in cam_list:
        ensure_detection_thread(cam)
    return render_template('cameras.html', cameras=cam_list)


@app.route('/cameras/add', methods=['POST'])
@login_required
def add_camera():
    cfg    = load_config()
    cams   = cfg.get('cameras', [])
    label  = request.form.get('label', '').strip() or f'Camera {len(cams)+1}'
    source = request.form.get('source', 'webcam')
    rtsp   = request.form.get('rtsp_url', '').strip()
    cam_id = f'cam{int(time.time())}'

    location_name = request.form.get('location_name', '').strip()
    floor         = request.form.get('floor', '').strip()
    building      = request.form.get('building', '').strip()
    lat_str       = request.form.get('location_lat', '').strip()
    lng_str       = request.form.get('location_lng', '').strip()
    location_lat  = float(lat_str) if lat_str else None
    location_lng  = float(lng_str) if lng_str else None

    new_cam = {
        'id': cam_id, 'label': label, 'source': source,
        'rtsp_url': rtsp, 'reconnect_attempts': 5,
        'hazard_enabled': (source == 'rtsp'),
        'location_name': location_name, 'floor': floor, 'building': building,
        'location_lat': location_lat, 'location_lng': location_lng,
    }
    cams.append(new_cam)
    cfg['cameras'] = cams
    save_config(cfg)

    try:
        db.upsert_camera(
            cam_id=cam_id, label=label, source=source, rtsp_url=rtsp,
            location_name=location_name, floor=floor, building=building,
            location_lat=location_lat, location_lng=location_lng,
        )
    except Exception as e:
        log.warning(f'upsert_camera error: {e}')

    ensure_detection_thread(new_cam)
    flash(f'Camera "{label}" added.', 'success')
    return redirect(url_for('cameras'))


@app.route('/cameras/remove/<cam_id>', methods=['POST'])
@login_required
def remove_camera(cam_id):
    cfg  = load_config()
    cams = [c for c in cfg.get('cameras', []) if c['id'] != cam_id]
    cfg['cameras'] = cams
    save_config(cfg)
    try:
        db.remove_camera(cam_id)
    except Exception as e:
        log.warning(f'remove_camera error: {e}')
    flash('Camera removed.', 'success')
    return redirect(url_for('cameras'))


@app.route('/cameras/edit/<cam_id>', methods=['POST'])
@login_required
def edit_camera(cam_id):
    cfg  = load_config()
    cams = cfg.get('cameras', [])
    for c in cams:
        if c['id'] == cam_id:
            c['label']         = request.form.get('label', c['label']).strip()
            c['source']        = request.form.get('source', c['source'])
            c['rtsp_url']      = request.form.get('rtsp_url', c.get('rtsp_url', '')).strip()
            c['location_name'] = request.form.get('location_name', c.get('location_name', '')).strip()
            c['floor']         = request.form.get('floor', c.get('floor', '')).strip()
            c['building']      = request.form.get('building', c.get('building', '')).strip()
            lat_str = request.form.get('location_lat', '').strip()
            lng_str = request.form.get('location_lng', '').strip()
            c['location_lat'] = float(lat_str) if lat_str else None
            c['location_lng'] = float(lng_str) if lng_str else None
            try:
                db.upsert_camera(cam_id=cam_id, **{k: c[k] for k in
                    ['label', 'source', 'rtsp_url', 'location_name',
                     'floor', 'building', 'location_lat', 'location_lng']})
            except Exception as e:
                log.warning(f'upsert_camera edit error: {e}')
            break
    cfg['cameras'] = cams
    save_config(cfg)
    with _start_lock:
        _started_cams.discard(cam_id)
    flash('Camera updated. Detection restarting...', 'success')
    return redirect(url_for('cameras'))


# ── Live video feed ────────────────────────────────────────────────────────

@app.route('/video_feed/<cam_id>')
@login_required
def video_feed(cam_id):
    def gen():
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, 'Connecting...', (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (91, 141, 238), 2)
        _, buf = cv2.imencode('.jpg', blank)
        blank_bytes = buf.tobytes()
        while True:
            with _state_lock:
                frame_bytes = _annotated_frames.get(cam_id)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + (frame_bytes or blank_bytes) + b'\r\n')
            time.sleep(0.04)
    return Response(stream_with_context(gen()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/enroll_feed')
@login_required
def enroll_feed():
    def gen():
        cfg      = load_config()
        cams     = cfg.get('cameras', [])
        feed_cam = cams[0]['id'] if cams else 'cam0'
        blank    = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf   = cv2.imencode('.jpg', blank)
        blank_bytes = buf.tobytes()
        while True:
            with _state_lock:
                frame_bytes = _annotated_frames.get(feed_cam)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + (frame_bytes or blank_bytes) + b'\r\n')
            time.sleep(0.04)
    return Response(stream_with_context(gen()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Detection settings ─────────────────────────────────────────────────────

@app.route('/detection', methods=['GET', 'POST'])
@login_required
def detection_settings():
    cfg = load_config()
    if request.method == 'POST':
        rec = cfg.get('recognition', {})
        rec['tolerance']    = round(float(request.form.get('tolerance',    0.55)), 2)
        rec['model']        = request.form.get('model', 'hog')
        rec['upsample']     = int(request.form.get('upsample', 1))
        rec['scale_factor'] = round(float(request.form.get('scale_factor', 0.75)), 2)
        cfg['recognition']  = rec
        save_config(cfg)
        global _recognizer, _recognizer_mtime
        with _recognizer_lock:
            _recognizer       = None
            _recognizer_mtime = 0
        flash('Detection settings saved and applied.', 'success')
        return redirect(url_for('detection_settings'))
    return render_template('detection_settings.html', cfg=cfg)


@app.route('/start_cctv', methods=['POST'])
@login_required
def start_cctv():
    flash('Detection is always running — no manual start needed.', 'info')
    return redirect(url_for('dashboard'))


@app.route('/stop_cctv', methods=['POST'])
@login_required
def stop_cctv():
    flash('Detection runs continuously. Restart the server to stop.', 'info')
    return redirect(url_for('dashboard'))


# ── API endpoints ──────────────────────────────────────────────────────────

@app.route('/api/stats')
@login_required
def api_stats():
    s = get_stats()
    s['cctv_running']    = True
    s['intruder_active'] = any(v.get('intruder') for v in _det_status.values())
    return jsonify(s)


@app.route('/api/recent')
@login_required
def api_recent():
    try:
        return jsonify(db.get_recent_events(limit=20))
    except Exception as e:
        log.warning(f'api_recent error: {e}')
        return jsonify([])


@app.route('/api/alert_state')
@login_required
def api_alert_state():
    return jsonify({
        'intruder':      any(v.get('intruder') for v in _det_status.values()),
        'unread_alerts': alerts_store.unread_count(),
    })


@app.route('/api/alerts')
@login_required
def api_alerts():
    return jsonify(alerts_store.get_alerts(limit=50))


@app.route('/api/encoding_status')
@login_required
def api_encoding_status():
    return jsonify(_encoding_status)


# ── Startup ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        db.init_schema()
    except Exception as e:
        log.error(f'MySQL init failed: {e}')
        sys.exit(1)

    cfg = load_config()

    portal_cfg = cfg.get('portal', {})
    if not portal_cfg.get('admin_email') or not portal_cfg.get('admin_password'):
        log.error('[AUTH] admin_email / admin_password missing from config.json')
        sys.exit(1)

    # Build face encodings if missing or empty
    _pkl_path    = 'face_encodings.pkl'
    _need_encode = False
    if not os.path.exists(_pkl_path):
        _need_encode = True
        log.warning('No face_encodings.pkl — all faces will show as UNKNOWN.')
    else:
        try:
            import pickle as _chk_pickle
            with open(_pkl_path, 'rb') as _chk_f:
                _chk_data = _chk_pickle.load(_chk_f)
            _enc_count = len(_chk_data.get('encodings', _chk_data.get('embeddings', [])))
            if _enc_count == 0:
                log.warning('face_encodings.pkl has 0 encodings — re-encoding...')
                _need_encode = True
            else:
                log.info(f'[STARTUP] face_encodings.pkl OK ({_enc_count} encodings)')
        except Exception as _chk_e:
            log.warning(f'face_encodings.pkl unreadable ({_chk_e}) — re-encoding...')
            _need_encode = True

    if _need_encode:
        pd = 'authorized_persons'
        if os.path.exists(pd) and any(os.scandir(pd)):
            log.info('Starting background face encoder...')
            _run_encoder_background(reason='startup check')
        else:
            log.warning('No persons enrolled yet — add people via the portal.')

    cameras_cfg = cfg.get('cameras', [])
    if not cameras_cfg:
        log.warning('No cameras configured — add one via the portal.')
    for cam in cameras_cfg:
        ensure_detection_thread(cam)

    log.info(f'[PORTAL] Starting — http://{portal_cfg["host"]}:{portal_cfg["port"]}')

    # ── gevent WSGIServer ─────────────────────────────────────────────────
    # FIX (v15): monkey.patch_all(thread=False, subprocess=False)
    #
    # thread=False    — keeps real OS threads for YOLO/dlib/OpenVINO inference
    # subprocess=False — keeps real subprocess.Popen for recog_worker.py;
    #                    gevent-patched Popen pipes cause LoopExit in the
    #                    concurrent.futures ThreadPool the gevent Hub uses.
    #
    # _GLOBAL_RECOG_Q was created before patching (stdlib queue.Queue) so the
    # dispatcher thread's blocking .get() never goes through gevent's event loop.
    try:
        from gevent import monkey as _gmonkey
        _gmonkey.patch_all(thread=False, subprocess=False)
        from gevent.pywsgi import WSGIServer as _GeventWSGI
        _srv = _GeventWSGI(
            (portal_cfg['host'], portal_cfg['port']),
            app,
            log=None,
            error_log=None,
        )
        log.info('[PORTAL] Using gevent WSGIServer (streaming-safe)')
        _srv.serve_forever()
    except ImportError:
        log.warning(
            '[PORTAL] gevent not installed — falling back to Flask dev server. '
            'Run: pip install gevent'
        )
        app.run(
            host=portal_cfg['host'],
            port=portal_cfg['port'],
            debug=False,
            use_reloader=False,
            threaded=True,
        )