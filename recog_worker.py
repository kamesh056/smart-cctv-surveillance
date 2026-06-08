"""
recog_worker.py  (v3 — NeuralGuard fixed edition)

Fixes vs v2:
  - FIX RELOAD: Supports a special RELOAD command from portal so the
    subprocess re-reads face_encodings.pkl immediately after a person is
    enrolled or deleted — without restarting the process.
  - FIX AUTO-RELOAD: Checks face_encodings.pkl mtime on every frame so
    if portal triggers a rebuild, the worker picks it up within one frame.
  - FIX NULL RECOGNIZER: If FaceRecognizer failed on startup (e.g. pkl not
    found yet), retries loading on every frame until it succeeds.
  - FIX IPC LENGTH SENTINEL: A 4-byte length of 0xFFFFFFFF is the RELOAD
    signal — worker reloads and replies with 0-length response.
  - FIX: All output to stderr only — never stdout (would corrupt IPC pipe).
"""

import sys
import os
import pickle
import struct
import logging
import time

# ── Environment setup BEFORE any other import ─────────────────────────────
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── On Windows: put stdin/stdout in binary mode immediately ───────────────
if sys.platform == 'win32':
    import msvcrt
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdin.fileno(),  os.O_BINARY)

# ── Logging — FILE only, never stdout ─────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [RECOG-WORKER] %(message)s',
    handlers=[logging.FileHandler('logs/recog_worker.log', encoding='utf-8')],
)
log = logging.getLogger(__name__)

# Special IPC sentinel: 4 bytes 0xFFFFFFFF = reload command
_RELOAD_SENTINEL = 0xFFFFFFFF
_PKL_PATH = 'face_encodings.pkl'


# ── IPC helpers ────────────────────────────────────────────────────────────

def _send(data: bytes) -> None:
    """Write length-prefixed frame to stdout (binary)."""
    try:
        sys.stdout.buffer.write(struct.pack('<I', len(data)) + data)
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        sys.exit(0)
    except Exception as e:
        log.error(f'_send error: {e}')


def _recv() -> bytes | None:
    """
    Read one length-prefixed frame from stdin.
    Returns b'' (empty bytes) for RELOAD sentinel.
    Returns None on EOF/error.
    """
    try:
        hdr = sys.stdin.buffer.read(4)
        if len(hdr) < 4:
            return None
        length = struct.unpack('<I', hdr)[0]
        if length == _RELOAD_SENTINEL:
            return b''   # special: reload signal, no body
        if length == 0 or length > 50 * 1024 * 1024:
            log.warning(f'_recv: suspicious length {length} — skipping')
            return None
        data = sys.stdin.buffer.read(length)
        return data if len(data) == length else None
    except Exception as e:
        log.error(f'_recv error: {e}')
        return None


# ── Mtime tracker ──────────────────────────────────────────────────────────

def _get_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.warning(f'recog_worker started PID={os.getpid()}')

    recognizer  = None
    last_mtime  = 0.0
    frame_count = 0

    def _try_load():
        """Attempt to load/reload FaceRecognizer. Returns new instance or None."""
        nonlocal last_mtime
        try:
            from face_recognizer import FaceRecognizer
            inst = FaceRecognizer()
            last_mtime = _get_mtime(_PKL_PATH)
            n = len(inst._known_encodings)
            log.warning(f'FaceRecognizer loaded OK ({n} encodings, mtime={last_mtime:.1f})')
            return inst
        except Exception as e:
            log.error(f'FaceRecognizer load failed: {e}')
            return None

    # Initial load attempt
    recognizer = _try_load()

    while True:
        try:
            raw = _recv()

            # EOF
            if raw is None:
                log.warning('recog_worker: EOF on stdin — exiting')
                break

            # RELOAD command (sentinel)
            if raw == b'':
                log.warning('[RECOG-WORKER] Reload command received — reloading encodings')
                recognizer = _try_load()
                # Send empty-result ACK so portal doesn't hang waiting
                _send(pickle.dumps([]))
                continue

            # If no recognizer yet, retry loading on every 30th frame
            frame_count += 1
            if recognizer is None and frame_count % 30 == 0:
                recognizer = _try_load()

            # Auto-reload when pkl mtime changes (person enrolled/deleted)
            if recognizer is not None:
                cur_mtime = _get_mtime(_PKL_PATH)
                if cur_mtime != last_mtime and cur_mtime > 0:
                    log.warning(f'[RECOG-WORKER] face_encodings.pkl changed — auto-reloading')
                    recognizer = _try_load()

            # Deserialise frame
            try:
                frame = pickle.loads(raw)
            except Exception as pe:
                log.error(f'pickle.loads failed: {pe}')
                _send(pickle.dumps([]))
                continue

            if recognizer is None or frame is None:
                _send(pickle.dumps([]))
                continue

            # Run recognition
            try:
                results = recognizer.recognize_faces(frame)
            except Exception as re:
                log.error(f'recognize_faces error: {re}')
                results = []

            _send(pickle.dumps(results))

        except Exception as e:
            log.error(f'recog_worker outer loop error: {e}')
            try:
                _send(pickle.dumps([]))
            except Exception:
                break

    log.warning('recog_worker exiting')


if __name__ == '__main__':
    main()
