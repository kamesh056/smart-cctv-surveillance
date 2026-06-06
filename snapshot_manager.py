"""
snapshot_manager.py  (v2)
Captures a SINGLE best-quality snapshot per intrusion event.
No video recording. Saves to snapshots/ with timestamp.
"""
import cv2, os, threading
from datetime import datetime
from pathlib import Path

SNAP_DIR = Path('snapshots')
SNAP_DIR.mkdir(exist_ok=True)


class SnapshotManager:

    def __init__(self):
        self._lock         = threading.Lock()
        self.last_snapshot = None   # path of most recent snapshot

    def capture(self, frame, label='UNKNOWN'):
        """
        Save a single JPEG snapshot.
        Returns the file path.
        """
        with self._lock:
            ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_lbl = label.replace(' ', '_')
            path     = SNAP_DIR / f'intruder_{safe_lbl}_{ts}.jpg'

            # Save at full resolution
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.last_snapshot = str(path)
            print(f'[SNAPSHOT] Saved: {path}')
            return str(path)

    def get_last(self):
        return self.last_snapshot
