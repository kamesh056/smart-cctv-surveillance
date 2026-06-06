"""
recog_worker.py — Face recognition subprocess worker.
Runs as a child process to isolate dlib/face_recognition segfaults
from the main Flask portal process.

Protocol (stdin/stdout, binary):
  Parent sends:  4-byte little-endian length + pickle(numpy frame)
  Child replies: 4-byte little-endian length + pickle(list of results)
"""
import sys, os, pickle, struct, logging

# Suppress OpenCV noise
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [RECOG] %(message)s',
    handlers=[logging.FileHandler('logs/recog_worker.log')]
)
log = logging.getLogger(__name__)

def _send(data: bytes):
    sys.stdout.buffer.write(struct.pack('<I', len(data)) + data)
    sys.stdout.buffer.flush()

def _recv() -> bytes | None:
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    length = struct.unpack('<I', hdr)[0]
    data = sys.stdin.buffer.read(length)
    return data if len(data) == length else None

def main():
    log.warning('recog_worker started')
    recognizer = None
    try:
        from face_recognizer import FaceRecognizer
        recognizer = FaceRecognizer()
        log.warning('FaceRecognizer loaded OK')
    except Exception as e:
        log.error(f'FaceRecognizer load failed: {e}')

    while True:
        try:
            raw = _recv()
            if raw is None:
                break
            frame = pickle.loads(raw)
            if recognizer is None:
                results = []
            else:
                results = recognizer.recognize_faces(frame)
            _send(pickle.dumps(results))
        except Exception as e:
            log.error(f'recog_worker error: {e}')
            _send(pickle.dumps([]))

if __name__ == '__main__':
    main()
