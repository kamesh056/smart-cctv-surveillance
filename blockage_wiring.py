"""
blockage_wiring.py
==================
One-stop helper: call `wire_blockage_alerts(streams, config_path)` right
after you call `load_cameras_from_config()` and this module does everything:

  1. Reads `blockage_detection` settings from config.json.
  2. Attaches on_blocked_callback and on_cleared_callback to every
     CameraStream so they fire automatically from the camera loop.
  3. Callbacks call AlertSystem to log portal alerts.
  4. Optionally saves a greyscale snapshot at the moment of blockage so
     operators can see what the camera looked like when it was covered.

Usage (in portal.py / startup.py):
---------------------------------------
    from camera_stream   import load_cameras_from_config
    from blockage_wiring import wire_blockage_alerts

    streams = load_cameras_from_config('config.json')
    wire_blockage_alerts(streams, config_path='config.json')

    for s in streams:
        s.start()
---------------------------------------
That's it. Alerts appear in /alerts automatically.
"""

import json
import os
import time
import cv2

from alert_system import AlertSystem

_alert_system = AlertSystem()


def wire_blockage_alerts(streams: list, config_path: str = 'config.json'):
    """
    Attach blockage / cleared callbacks to every CameraStream in `streams`.

    Parameters
    ----------
    streams     : list of CameraStream objects (from load_cameras_from_config)
    config_path : path to config.json (to read snapshot dir + enabled flag)
    """
    with open(config_path) as f:
        cfg = json.load(f)

    bcfg          = cfg.get('blockage_detection', {})
    enabled       = bcfg.get('enabled', True)
    save_snapshot = bcfg.get('save_snapshot', True)
    alert_on_clear= bcfg.get('alert_on_clear', True)
    snapshot_dir  = cfg.get('snapshot_dir', 'snapshots')

    if not enabled:
        print('[BLOCKAGE] Blockage detection is DISABLED in config.json.')
        return

    os.makedirs(snapshot_dir, exist_ok=True)

    for stream in streams:
        # Capture loop vars per stream with default-arg trick
        def _make_blocked_cb(s, snap_dir, do_snap):
            def _on_blocked(camera_label: str, reason: str, status: dict):
                snap_path = None
                if do_snap:
                    frame = s.get_frame()
                    if frame is not None:
                        ts   = time.strftime('%Y%m%d_%H%M%S')
                        name = f'blocked_{s.cam_id or camera_label}_{ts}.jpg'
                        snap_path = os.path.join(snap_dir, name)
                        try:
                            cv2.imwrite(snap_path, frame)
                        except Exception as e:
                            print(f'[BLOCKAGE] snapshot save failed: {e}')
                            snap_path = None

                _alert_system.send_camera_blocked_alert(
                    camera        = camera_label,
                    reason        = reason,
                    snapshot_path = snap_path,
                )
            return _on_blocked

        def _make_cleared_cb(s, do_alert):
            def _on_cleared(camera_label: str, status: dict):
                if do_alert:
                    _alert_system.send_camera_cleared_alert(camera=camera_label)
            return _on_cleared

        stream.on_blocked_callback = _make_blocked_cb(stream, snapshot_dir, save_snapshot)
        stream.on_cleared_callback = _make_cleared_cb(stream, alert_on_clear)

        print(f'[BLOCKAGE] Wired blockage detection → {stream.label}')
