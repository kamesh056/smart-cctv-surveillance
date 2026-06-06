"""
access_logger.py  (v3 — MySQL)
Writes ONE entry per person visit: entry time and exit time only.
Now backed by MySQL via db_mysql.py instead of SQLite.
"""
import threading
from datetime import datetime
from pathlib import Path

import db_mysql as db

LOG_DIR  = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'access_log.txt'


class AccessLogger:

    def __init__(self):
        self._lock     = threading.Lock()
        # Track active sessions: key = person_id or "UNKNOWN_<name>", value = MySQL row id
        self._sessions = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _now(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _append_txt(self, line):
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    # ── Public API ─────────────────────────────────────────────────────────

    def record_entry(self, person_name, person_id=None,
                     status='AUTHORIZED', snapshot=None,
                     camera_id=None, camera='webcam', location=None):
        """
        Call when a person first appears in the frame.
        Returns the session key so you can later call record_exit().
        """
        with self._lock:
            key = person_id or f'UNKNOWN_{person_name}'
            if key in self._sessions:
                return key   # already open — deduplicate

            row_id = db.record_entry(
                person_name=person_name,
                person_id=person_id,
                status=status,
                snapshot=snapshot,
                camera_id=camera_id,
                camera_label=camera,
                location=location,
            )
            self._sessions[key] = row_id

            icon = '✅' if status == 'AUTHORIZED' else '🚨'
            ts   = self._now()
            line = (f'{icon} ENTRY  | {ts} | {status:14s} | '
                    f'{person_name:20s} | Camera: {camera}')
            if snapshot:
                line += f' | Snapshot: {snapshot}'
            self._append_txt(line)
            print(f'[LOG] {line}')

            # Also write to MySQL system log
            db.log_system('INFO', line, module='access_logger')

            return key

    def record_exit(self, session_key):
        """
        Call when the person leaves the frame.
        Updates the MySQL row with exit time and duration.
        """
        with self._lock:
            if session_key not in self._sessions:
                return
            row_id = self._sessions.pop(session_key)
            db.record_exit(row_id)

            ts   = self._now()
            line = f'EXIT   | {ts} | session_key={session_key}'
            self._append_txt(line)
            print(f'[LOG] {line}')

    def get_recent_events(self, limit=100):
        """Return recent events for the web portal."""
        return db.get_recent_events(limit=limit)
