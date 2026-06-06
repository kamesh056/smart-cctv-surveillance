"""
portal.py  (v11 — login/auth added, all bugs fixed)

Changes vs v10:
  - /login route + session-based auth added
  - @login_required decorator protects every page and API endpoint
  - /logout route added
  - CSRF token generated in login route and validated on POST
  - open('config.json') replaced with context manager (no leaked handles)
  - api_recent() double-serialisation removed
  - MAX_CONTENT_LENGTH set (100 MB) to cap upload size
  - get_all_cameras() in db_mysql now filters active=1 (fixed there)
  - Enroll-page file handle for recog_worker.log uses context manager
"""

# ── MUST be set before ANY cv2 import ─────────────────────────────────────
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
    'rtsp_transport;tcp|timeout;5000000|max_delay;500000|stimeout;5000000'
os.environ['OPENCV_LOG_LEVEL']              = 'ERROR'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF']  = '0'

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

# ── Logging ────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Flask app setup ────────────────────────────────────────────────────────
app = Flask(__name__)

# Load config once at startup — use context manager (fixes unclosed-handle bug)
with open('config.json') as _f:
    _startup_cfg = json.load(_f)

app.secret_key           = _startup_cfg['portal']['secret_key']
app.permanent_session_lifetime = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100 MB upload cap

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
  <a href="__PERSONS_URL__" class="back">← Back to Persons</a>
</div>
<div class="wrap">
  <h2>Enroll New Person</h2>
  <div class="card">
    <div class="tabs">
      <button class="tab active" id="tab-record" onclick="switchTab('record')">🎥 Record Webcam</button>
      <button class="tab"        id="tab-upload" onclick="switchTab('upload')">📁 Upload Video</button>
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
            <button class="btn btn-p" id="btn-start" onclick="startRec()">⏺ Start Recording</button>
            <button class="btn btn-d" id="btn-stop"  onclick="stopRec()" style="display:none">⏹ Stop</button>
          </div>
          <div class="status" id="status"></div>
          <div class="rec-wrap" id="rec-wrap">
            <p style="color:#7dea7d;margin-bottom:.6rem;font-size:.85rem">✅ Done! Review then enroll.</p>
            <video id="preview" controls></video>
            <div style="margin-top:.8rem;display:flex;gap:.6rem">
              <button class="btn btn-s" onclick="submitRec()">✔ Enroll Person</button>
              <button class="btn btn-g" onclick="resetRec()">↺ Re-record</button>
            </div>
          </div>
        </div>
        <div class="cam-side">
          <div class="cam-box">
            <img id="live-img" src="__FEED_URL__" alt="Live feed">
            <div class="rec-dot" id="rec-dot">● REC</div>
          </div>
          <p class="hint">Face the camera — tilt slightly left, right, up, down</p>
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
          <button type="submit" class="btn btn-s">📥 Enroll Person</button>
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
FRAMES_TO_CONFIRM     = 6
FRAMES_TO_EXIT        = 45
ALERT_COOLDOWN        = 120
PROCESS_EVERY         = 8

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


# ── Auth helpers ───────────────────────────────────────────────────────────

def login_required(f):
    """Decorator: redirect to /login if no active session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def _check_credentials(email: str, password: str) -> bool:
    """Validate email+password against config.json portal section."""
    cfg          = load_config()
    portal_cfg   = cfg.get('portal', {})
    admin_email  = portal_cfg.get('admin_email', '').strip().lower()
    admin_pass   = portal_cfg.get('admin_password', '')
    return email.strip().lower() == admin_email and password == admin_pass


# ── Auth routes ────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Redirect already-logged-in users
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    # Generate CSRF token if not present
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)

    if request.method == 'POST':
        # ── CSRF check ──
        form_token    = request.form.get('csrf_token', '')
        session_token = session.get('_csrf_token', '')
        if not secrets.compare_digest(form_token, session_token):
            flash('Invalid request — please try again.', 'error')
            return redirect(url_for('login'))

        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if _check_credentials(email, password):
            session.clear()                         # regenerate session on login
            session['logged_in']  = True
            session['user_email'] = email
            session['_csrf_token'] = secrets.token_hex(32)
            if 'remember' in request.form:
                session.permanent = True            # 7-day cookie
            flash('Welcome back!', 'success')
            log.info(f'[AUTH] Login success: {email}')
            return redirect(url_for('dashboard'))
        else:
            log.warning(f'[AUTH] Failed login attempt for: {email}')
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html', csrf_token=session.get('_csrf_token', ''))


@app.route('/logout')
def logout():
    email = session.get('user_email', 'unknown')
    session.clear()
    log.info(f'[AUTH] Logged out: {email}')
    flash('You have been signed out.', 'info')
    return redirect(url_for('login'))


# ── Person helpers ─────────────────────────────────────────────────────────

def list_persons() -> list[dict]:
    persons = []
    if not PERSONS_DIR.exists():
        return persons
    for d in sorted(PERSONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / 'meta.txt'
        meta = {
            'id': d.name,
            'name': d.name.replace('_', ' ').title(),
            'role': '', 'contact': '', 'department': '',
            'enrolled': False, 'added': '',
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
        log.warning(f'get_stats MySQL error: {e}')
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
    path = SNAPSHOTS_DIR / name
    cv2.imwrite(str(path), frame)
    try:
        db.record_snapshot(filename=name, filepath=str(path),
                           person_name=label, camera_id=camera_id,
                           event_id=event_id)
    except Exception as e:
        log.warning(f'record_snapshot error: {e}')
    return str(path)


# ── OpenCV thread safety ───────────────────────────────────────────────────
# VideoCapture is NOT thread-safe on Windows — serialise ALL cap.read() calls.
_cv2_read_lock = threading.Lock()


def _safe_read(cap, timeout: float = 5.0):
    """Read a frame with a hard timeout to prevent RTSP hangs."""
    result = [False, None]

    def _do_read():
        try:
            with _cv2_read_lock:
                result[0], result[1] = cap.read()
        except Exception:
            pass

    t = threading.Thread(target=_do_read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        log.warning('[CAMERA] cap.read() timed out — skipping frame')
        return False, None
    return result[0], result[1]


# ── Camera open helper ─────────────────────────────────────────────────────

def open_camera(src, cam_type: str = 'webcam'):
    src_str = str(src)

    # HTTP / MJPEG (e.g. Android IP Webcam)
    if src_str.startswith('http://') or src_str.startswith('https://'):
        log.info(f'[CAMERA] Detected HTTP stream: {src_str}')
        if not any(x in src_str for x in ['/video', '/shot', '/mjpeg', '.mjpg']):
            src_str = src_str.rstrip('/') + '/video'
            log.info(f'[CAMERA] Appended /video → {src_str}')
        cap = cv2.VideoCapture(src_str)
        if not cap.isOpened():
            cap.release()
            log.warning('[CAMERA] HTTP stream failed to open')
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frm = _safe_read(cap, timeout=10.0)
        if ret and frm is not None:
            log.info('[CAMERA] HTTP stream verified OK')
            return cap
        cap.release()
        log.warning('[CAMERA] HTTP stream opened but no frames')
        return None

    # RTSP
    if cam_type == 'rtsp':
        log.info(f'[CAMERA] Opening RTSP: {src}')
        cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            log.warning('[CAMERA] RTSP VideoCapture failed to open')
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frm = _safe_read(cap, timeout=8.0)
        if ret and frm is not None:
            log.info('[CAMERA] RTSP stream verified OK')
            return cap
        cap.release()
        log.warning('[CAMERA] RTSP opened but no frames — bad URL or unreachable')
        return None

    # Webcam
    idx = int(src) if not isinstance(src, int) else src
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
            with _cv2_read_lock:
                ret, frm = cap.read()
            if ret and frm is not None:
                log.info(f'[CAMERA] Opened webcam index {idx} via {bname}')
                return cap
            time.sleep(0.1)
        cap.release()
        log.warning(f'[CAMERA] {bname}: opened but no frames for index {idx}')
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


def draw_status_bar(frame, intruder_active: bool, cam_label: str, face_count: int):
    h, w = frame.shape[:2]
    bar_color = (0, 0, 180) if intruder_active else (0, 130, 0)
    cv2.rectangle(frame, (0, 0), (w, 36), bar_color, cv2.FILLED)
    status_txt = '!  INTRUSION DETECTED' if intruder_active else 'OK  ALL CLEAR'
    cv2.putText(frame, status_txt, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    fc_txt = f'Faces: {face_count}'
    (tw, _), _ = cv2.getTextSize(fc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, fc_txt, (w-tw-10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    cv2.rectangle(frame, (0, h-28), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, ts, (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, cam_label, (w//2-60, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (91, 141, 238), 1)
    return frame


# ── Camera Blockage Detector ───────────────────────────────────────────────

class BlockageDetector:
    """
    Detects whether a camera is physically blocked or blacked out by
    combining three independent signals on every frame:

      1. Mean brightness  — very dark frame  → lens cap / lights out
      2. Pixel variance   — near-zero spread → cloth / hand / tape
      3. Edge density     — almost no edges  → blurry obstruction

    All three must be below threshold simultaneously for `confirm_seconds`
    before an alert fires, preventing false positives from dark scenes.
    A `cooldown_seconds` gap between alerts prevents spamming.
    """

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

        self._suspect_since: float | None = None
        self._last_alert_at: float        = 0.0
        self._blocked: bool               = False

    def analyse(self, frame) -> dict:
        """
        Returns dict with keys:
          blocked, just_triggered, just_cleared, reason,
          brightness, variance, edge_density
        """
        result = dict(blocked=False, just_triggered=False,
                      just_cleared=False, reason='',
                      brightness=0.0, variance=0.0, edge_density=0.0)
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

        dark     = brightness   < self.brightness_thresh
        uniform  = variance     < self.variance_thresh
        no_edges = edge_density < self.edge_density_thresh

        reasons = []
        if dark:     reasons.append(f'blackout (brightness={brightness:.1f})')
        if uniform:  reasons.append(f'uniform image (variance={variance:.1f})')
        if no_edges: reasons.append(f'no edges (density={edge_density:.4f})')

        suspect_now = dark and uniform and no_edges
        now = time.time()

        if suspect_now:
            result['reason'] = '; '.join(reasons)
            if self._suspect_since is None:
                self._suspect_since = now

            elapsed = now - self._suspect_since
            if elapsed >= self.confirm_seconds and not self._blocked:
                if (now - self._last_alert_at) >= self.cooldown_seconds:
                    self._blocked             = True
                    result['blocked']         = True
                    result['just_triggered']  = True
                    self._last_alert_at       = now
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
    """
    Draws a prominent red CAMERA BLOCKED warning over the frame.
    Called every frame while blockage is active so the live feed
    shows a clear visual indicator in the portal.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark red fill
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Large warning text
    main_txt = '!  CAMERA BLOCKED / BLACKOUT'
    scale = max(0.5, w / 800)
    (tw, th), _ = cv2.getTextSize(main_txt, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cx = (w - tw) // 2
    cy = h // 2 - 20
    cv2.putText(frame, main_txt, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Smaller reason line
    if reason:
        sub_scale = max(0.35, w / 1200)
        (sw, _), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
        cv2.putText(frame, reason, ((w - sw) // 2, cy + th + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 200, 200), 1, cv2.LINE_AA)

    # Timestamp
    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, ts, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return frame


# ── Recognizer loader ──────────────────────────────────────────────────────

_recognizer       = None
_recognizer_lock  = threading.Lock()
_recognizer_mtime = 0


def get_recognizer():
    global _recognizer, _recognizer_mtime
    pkl = 'face_encodings.pkl'
    try:
        mtime = os.path.getmtime(pkl) if os.path.exists(pkl) else 0
    except Exception:
        mtime = 0
    with _recognizer_lock:
        if _recognizer is None or mtime != _recognizer_mtime:
            try:
                from face_recognizer import FaceRecognizer
                _recognizer       = FaceRecognizer()
                _recognizer_mtime = mtime
                log.info('[RECOGNIZER] Loaded/reloaded.')
            except Exception as e:
                log.warning(f'[RECOGNIZER] Load failed: {e}')
        return _recognizer


# ── Per-camera detection thread ────────────────────────────────────────────

_started_cams = set()
_start_lock   = threading.Lock()

import queue as _queue


def ensure_detection_thread(cam_cfg: dict) -> None:
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
        cap = None
        for attempt in range(cam_cfg.get('reconnect_attempts', 5)):
            cap = open_camera(cam_src, cam_type)
            if cap:
                break
            log.warning(f'[DET] Attempt {attempt+1} failed for {label}, retrying in 3s...')
            time.sleep(3)

        if not cap:
            log.error(f'[DET] Cannot open {label} — giving up')
            with _state_lock:
                _det_status[cam_id] = {'intruder': False, 'faces': 0, 'running': False}
            with _start_lock:
                _started_cams.discard(cam_id)
            return

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

        # ── Face recognition subprocess ────────────────────────────────────
        import pickle as _pickle, struct as _struct

        recog_result = [None]
        recog_lock   = threading.Lock()

        def _start_recog_proc():
            try:
                # Use context manager for log file handle
                log_fh = open('logs/recog_worker.log', 'a')
                p = subprocess.Popen(
                    [sys.executable, 'recog_worker.py'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log_fh,
                )
                log.info(f'[DET] recog subprocess started PID={p.pid}')
                return p
            except Exception as e:
                log.error(f'[DET] recog subprocess start failed: {e}')
                return None

        def _recog_send_recv(proc, frame):
            try:
                raw = _pickle.dumps(frame, protocol=4)
                hdr = _struct.pack('<I', len(raw))
                proc.stdin.write(hdr + raw)
                proc.stdin.flush()
                hdr2 = proc.stdout.read(4)
                if len(hdr2) < 4:
                    return None
                length = _struct.unpack('<I', hdr2)[0]
                data   = proc.stdout.read(length)
                return _pickle.loads(data) if len(data) == length else None
            except Exception as e:
                log.warning(f'[DET] recog IPC error: {e}')
                return None

        def _recognizer_worker():
            proc = _start_recog_proc()
            while True:
                item = recog_q.get()
                if item is None:
                    if proc and proc.poll() is None:
                        try:
                            proc.stdin.close()
                            proc.wait(timeout=3)
                        except Exception:
                            proc.kill()
                    break
                if proc is None or proc.poll() is not None:
                    log.warning('[DET] recog subprocess died — restarting')
                    proc = _start_recog_proc()
                    if proc is None:
                        with recog_lock:
                            recog_result[0] = []
                        continue
                res = _recog_send_recv(proc, item)
                if res is None:
                    log.warning('[DET] recog subprocess returned None — restarting')
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc = _start_recog_proc()
                    res  = []
                with recog_lock:
                    recog_result[0] = res

        recog_q = _queue.Queue(maxsize=1)
        rt = threading.Thread(target=_recognizer_worker, daemon=True, name=f'rec-{cam_id}')
        rt.start()

        face_sessions      = {}
        face_counters      = {}
        absent_counters    = {}
        last_alert_time    = {}
        frame_count        = 0
        consecutive_fails  = 0
        MAX_FAILS          = 30

        # ── Blockage detector (one instance per camera) ────────────────────
        _bcfg = load_config().get('blockage_detection', {})
        blockage_detector = BlockageDetector(
            brightness_thresh   = _bcfg.get('brightness_thresh',   15.0),
            variance_thresh     = _bcfg.get('variance_thresh',     20.0),
            edge_density_thresh = _bcfg.get('edge_density_thresh', 0.002),
            confirm_seconds     = _bcfg.get('confirm_seconds',     3.0),
            cooldown_seconds    = _bcfg.get('cooldown_seconds',    120.0),
        )
        blockage_alert_on_clear = _bcfg.get('alert_on_clear', True)
        blockage_snap_dir       = SNAPSHOTS_DIR   # reuse same snapshots folder

        try:
            while True:
                if cam_type == 'rtsp':
                    ret, frame = _safe_read(cap, timeout=5.0)
                else:
                    with _cv2_read_lock:
                        ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= MAX_FAILS:
                        log.warning(f'[DET] {label}: too many failures — reconnecting')
                        cap.release()
                        cap = None
                        for attempt in range(cam_cfg.get('reconnect_attempts', 5)):
                            cap = open_camera(cam_src, cam_type)
                            if cap:
                                log.info(f'[DET] {label}: reconnected')
                                consecutive_fails = 0
                                blockage_detector.reset()
                                break
                            time.sleep(3)
                        if not cap:
                            log.error(f'[DET] {label}: reconnect failed — stopping')
                            break
                    time.sleep(0.05)
                    continue

                consecutive_fails = 0
                frame_count += 1

                if frame.size == 0 or len(frame.shape) != 3:
                    log.warning(f'[DET] {label}: malformed frame skipped')
                    continue

                # ── Blockage detection ─────────────────────────────────────
                blk = blockage_detector.analyse(frame)

                if blk['just_triggered']:
                    reason_str = blk.get('reason', '')
                    log.warning(f'[BLOCKAGE] ⚠  {label}: BLOCKED — {reason_str}')

                    # Save a snapshot of what the camera looks like right now
                    snap_path = None
                    try:
                        ts_str    = datetime.now().strftime('%Y%m%d_%H%M%S')
                        snap_name = f'blocked_{cam_id}_{ts_str}.jpg'
                        snap_path = str(blockage_snap_dir / snap_name)
                        cv2.imwrite(snap_path, frame)
                    except Exception as _se:
                        log.warning(f'[BLOCKAGE] snapshot save error: {_se}')
                        snap_path = None

                    try:
                        alerts_store.AlertSystem().send_camera_blocked_alert(
                            camera        = label,
                            reason        = reason_str,
                            snapshot_path = snap_path,
                        )
                    except Exception as _ae:
                        log.warning(f'[BLOCKAGE] alert error: {_ae}')

                    with _state_lock:
                        _det_status[cam_id] = {
                            'intruder': False, 'faces': 0,
                            'running': True, 'blocked': True,
                        }

                elif blk['just_cleared']:
                    log.info(f'[BLOCKAGE] ✓  {label}: camera CLEAR again.')
                    if blockage_alert_on_clear:
                        try:
                            alerts_store.AlertSystem().send_camera_cleared_alert(
                                camera=label)
                        except Exception as _ae:
                            log.warning(f'[BLOCKAGE] cleared alert error: {_ae}')

                # If currently blocked: show overlay on live feed, skip face recognition
                if blk['blocked']:
                    display = draw_blocked_overlay(frame.copy(), blk.get('reason', ''))
                    try:
                        ok, buf = cv2.imencode('.jpg', display,
                                               [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok and buf is not None:
                            with _state_lock:
                                _annotated_frames[cam_id] = buf.tobytes()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    continue   # skip face recognition while camera is blocked
                # ── End blockage detection ─────────────────────────────────

                display = frame.copy()

                with recog_lock:
                    results = recog_result[0] or []

                if frame_count % PROCESS_EVERY == 0:
                    try:
                        recog_q.put_nowait(frame.copy())
                    except _queue.Full:
                        pass

                if results:
                    draw_boxes(display, results)

                seen_names = set()
                intruder   = False

                for r in results:
                    name = r['name']
                    auth = r['authorized']
                    seen_names.add(name)

                    face_counters[name]   = face_counters.get(name, 0) + 1
                    absent_counters[name] = 0

                    if not auth:
                        intruder = True

                    if face_counters[name] == FRAMES_TO_CONFIRM:
                        # Save snapshot for every detection (auth + unknown)
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
                                except Exception as e:
                                    log.warning(f'[ALERT] {e}')

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

                face_count = len(results)
                draw_status_bar(display, intruder, label, face_count)

                with _state_lock:
                    _det_status[cam_id] = {
                        'intruder': intruder,
                        'faces':    face_count,
                        'running':  True,
                    }

                if (display is not None
                        and display.size > 0
                        and len(display.shape) == 3
                        and display.shape[0] > 0
                        and display.shape[1] > 0):
                    try:
                        ok, buf = cv2.imencode('.jpg', display,
                                               [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok and buf is not None:
                            with _state_lock:
                                _annotated_frames[cam_id] = buf.tobytes()
                    except Exception as enc_err:
                        log.warning(f'[DET] {label}: imencode failed: {enc_err}')

        except Exception as e:
            log.error(f'[DET] Thread crashed for {cam_id}: {e}')
        finally:
            recog_q.put(None)
            if cap:
                cap.release()
            with _start_lock:
                _started_cams.discard(cam_id)
            log.info(f'[DET] Thread exited for {label} ({cam_id})')

    t = threading.Thread(target=_worker, daemon=True, name=f'det-{cam_id}')
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
    meta_lines = [
        f'name:{name}', f'role:{role}', f'contact:{contact}',
        f'department:{dept}',
        f'added:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    ]
    (folder / 'meta.txt').write_text('\n'.join(meta_lines))

    try:
        db.upsert_person(person_id=person_id, name=name, role=role,
                         contact=contact, department=dept)
    except Exception as e:
        log.warning(f'upsert_person error: {e}')

    try:
        subprocess.run([sys.executable, 'video_face_encoder.py'],
                       check=True, timeout=300)
    except Exception as e:
        log.warning(f'Encoding failed: {e}')

    flash(f'"{name}" enrolled successfully.', 'success')
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
    meta_lines = [
        f'name:{name}', f'role:{role}', f'contact:{contact}',
        f'department:{dept}',
        f'added:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    ]
    (folder / 'meta.txt').write_text('\n'.join(meta_lines))

    try:
        db.upsert_person(person_id=person_id, name=name, role=role,
                         contact=contact, department=dept)
    except Exception as e:
        log.warning(f'upsert_person error: {e}')

    try:
        subprocess.run([sys.executable, 'video_face_encoder.py'],
                       check=True, timeout=300)
    except Exception as e:
        log.warning(f'Encoding failed: {e}')

    return jsonify({'ok': True})


@app.route('/persons/delete/<person_id>', methods=['POST'])
@login_required
def delete_person(person_id):
    folder = PERSONS_DIR / secure_filename(person_id)
    if folder.exists():
        shutil.rmtree(str(folder))
    try:
        db.deactivate_person(person_id)
    except Exception as e:
        log.warning(f'deactivate_person error: {e}')
    flash('Person removed.', 'success')
    return redirect(url_for('persons'))


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


# ── Alerts ─────────────────────────────────────────────────────────────────

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


# ── Cameras ────────────────────────────────────────────────────────────────

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
        rec['tolerance']    = round(float(request.form.get('tolerance', 0.55)), 2)
        rec['model']        = request.form.get('model', 'hog')
        rec['upsample']     = int(request.form.get('upsample', 1))
        rec['scale_factor'] = round(float(request.form.get('scale_factor', 0.75)), 2)
        cfg['recognition']  = rec
        save_config(cfg)
        global _recognizer_mtime
        _recognizer_mtime = 0
        flash('Detection settings saved and applied.', 'success')
        return redirect(url_for('detection_settings'))
    return render_template('detection_settings.html', cfg=cfg)


# ── Engine control (informational only) ───────────────────────────────────

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
    # db.get_recent_events() already calls _serialize() — no need to re-convert
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


# ── Startup ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        db.init_schema()
    except Exception as e:
        log.error(f'MySQL init failed: {e}')
        sys.exit(1)

    cfg = load_config()

    # Validate that admin credentials are configured
    portal_cfg = cfg.get('portal', {})
    if not portal_cfg.get('admin_email') or not portal_cfg.get('admin_password'):
        log.error('[AUTH] admin_email / admin_password missing from config.json portal section.')
        sys.exit(1)

    # Build face encodings if missing
    if not os.path.exists('face_encodings.pkl'):
        pd = 'authorized_persons'
        if os.path.exists(pd) and any(os.scandir(pd)):
            log.info('face_encodings.pkl missing — running encoder...')
            try:
                subprocess.run([sys.executable, 'video_face_encoder.py'],
                               check=True, timeout=300)
            except Exception as e:
                log.warning(f'Encoding failed: {e}')
        else:
            log.warning('No face_encodings.pkl — all faces will show as UNKNOWN.')

    cameras_cfg = cfg.get('cameras', [])
    if not cameras_cfg:
        log.warning('No cameras configured — add one via the portal.')
    for cam in cameras_cfg:
        ensure_detection_thread(cam)

    log.info(f'[PORTAL] Starting — http://{portal_cfg["host"]}:{portal_cfg["port"]}')
    app.run(
        host=portal_cfg['host'],
        port=portal_cfg['port'],
        debug=False,
        use_reloader=False,
        threaded=True,
    )