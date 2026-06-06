"""
alert_system.py — In-memory alert store for NeuralGuard portal.
All intrusion alerts are stored here and surfaced in the /alerts page.
No email is sent — portal-only.
"""

import threading
from datetime import datetime

_lock   = threading.Lock()
_alerts = []          # list of dicts, newest first
_unread = 0


class AlertSystem:
    """Thin wrapper so portal.py can call AlertSystem().send_intrusion_alert(...)"""

    def send_intrusion_alert(self, snapshot_path: str, camera: str = 'Unknown'):
        """Record an intrusion alert in memory."""
        _add_alert(
            title=f'Intrusion Detected — {camera}',
            message=f'Unknown person detected on camera "{camera}".',
            snapshot=snapshot_path,
            camera=camera,
            level='danger',
        )

    def send_camera_blocked_alert(self, camera: str = 'Unknown',
                                  reason: str = '',
                                  snapshot_path: str = None):
        """
        Record a camera-blockage alert.

        Called automatically by CameraStream's on_blocked_callback.
        `reason` is a human-readable string built by BlockageDetector
        (e.g. 'blackout (brightness=3.2); no edges (density=0.0001)').
        """
        detail = f' Reason: {reason}.' if reason else ''
        _add_alert(
            title   = f'Camera Blocked / Blackout — {camera}',
            message = (
                f'Camera "{camera}" appears to be physically blocked or blacked out.{detail} '
                f'Please check the camera immediately.'
            ),
            snapshot = snapshot_path,
            camera   = camera,
            level    = 'warning',          # orange badge — distinct from red intrusion
        )

    def send_camera_cleared_alert(self, camera: str = 'Unknown'):
        """
        Record a camera-restored info alert so operators know it's back.
        """
        _add_alert(
            title   = f'Camera Restored — {camera}',
            message = f'Camera "{camera}" feed has returned to normal.',
            snapshot = None,
            camera   = camera,
            level    = 'info',
        )


def _add_alert(title: str, message: str, snapshot: str = None,
               camera: str = '', level: str = 'danger'):
    global _unread
    alert = {
        'id':       id(object()),          # unique enough for in-memory use
        'title':    title,
        'message':  message,
        'snapshot': snapshot,
        'camera':   camera,
        'level':    level,                 # 'danger' | 'warning' | 'info'
        'ts':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'read':     False,
    }
    with _lock:
        _alerts.insert(0, alert)
        if len(_alerts) > 500:             # cap at 500 entries
            _alerts.pop()
        _unread += 1


def get_alerts(limit: int = 50):
    with _lock:
        return list(_alerts[:limit])


def unread_count() -> int:
    with _lock:
        return _unread


def mark_all_read():
    global _unread
    with _lock:
        for a in _alerts:
            a['read'] = True
        _unread = 0


def clear_alerts():
    global _unread
    with _lock:
        _alerts.clear()
        _unread = 0
