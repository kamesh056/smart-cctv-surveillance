"""
db_mysql.py — Centralised MySQL connection + schema management for NeuralGuard.

All other modules import from here instead of using sqlite3 directly.
Uses mysql-connector-python (pip install mysql-connector-python).

Tables created:
  cameras        — registered camera info + location
  persons        — enrolled authorised persons
  access_events  — entry/exit log with duration
  snapshots      — snapshot file references
  system_logs    — general system messages
"""

import mysql.connector
from mysql.connector import pooling
from datetime import datetime
import json, os, threading
from dotenv import load_dotenv

load_dotenv()

# ── Connection settings (override via .env or config.json) ─────────────────

def _get_cfg():
    """Read MySQL credentials from environment or config.json."""
    cfg_path = 'config.json'
    db_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            data = json.load(f)
        db_cfg = data.get('mysql', {})

    return {
        'host':     os.getenv('MYSQL_HOST',     db_cfg.get('host',     'localhost')),
        'port':     int(os.getenv('MYSQL_PORT', db_cfg.get('port',     3306))),
        'user':     os.getenv('MYSQL_USER',     db_cfg.get('user',     'neuralguard')),
        'password': os.getenv('MYSQL_PASSWORD', db_cfg.get('password', '')),
        'database': os.getenv('MYSQL_DATABASE', db_cfg.get('database', 'neuralguard')),
    }


# ── Connection pool (created once at import time) ──────────────────────────

_pool      = None
_pool_lock = threading.Lock()


def _make_pool():
    global _pool
    cfg = _get_cfg()
    _pool = pooling.MySQLConnectionPool(
        pool_name='neuralguard_pool',
        pool_size=10,
        host=cfg['host'],
        port=cfg['port'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        autocommit=False,
        connection_timeout=10,
    )
    print(f"[DB] Connected to MySQL at {cfg['host']}:{cfg['port']} "
          f"— database '{cfg['database']}'")


def get_connection():
    """Return a pooled MySQL connection. Initialises pool on first call."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _make_pool()
    return _pool.get_connection()


# ── Schema ─────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Cameras table: one row per physical camera
CREATE TABLE IF NOT EXISTS cameras (
    id                VARCHAR(64)  PRIMARY KEY,
    label             VARCHAR(128) NOT NULL,
    source            VARCHAR(16)  NOT NULL DEFAULT 'webcam',
    rtsp_url          VARCHAR(512),
    location_name     VARCHAR(128),      -- e.g. "Main Gate", "Server Room"
    location_lat      DECIMAL(10,7),     -- GPS latitude  (optional)
    location_lng      DECIMAL(10,7),     -- GPS longitude (optional)
    floor             VARCHAR(32),       -- e.g. "Ground Floor", "Floor 2"
    building          VARCHAR(64),
    reconnect_attempts INT DEFAULT 5,
    active            TINYINT(1) DEFAULT 1,
    added_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Persons table: authorised enrolled individuals
CREATE TABLE IF NOT EXISTS persons (
    id           VARCHAR(128) PRIMARY KEY,   -- folder name, e.g. "john_doe_123"
    name         VARCHAR(128) NOT NULL,
    role         VARCHAR(64),
    contact      VARCHAR(128),
    department   VARCHAR(64),
    access_level VARCHAR(32) DEFAULT 'standard',
    enrolled_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    active       TINYINT(1) DEFAULT 1,
    notes        TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Access events: one row per visit (entry → exit)
CREATE TABLE IF NOT EXISTS access_events (
    id           BIGINT       AUTO_INCREMENT PRIMARY KEY,
    person_name  VARCHAR(128) NOT NULL,
    person_id    VARCHAR(128),
    status       ENUM('AUTHORIZED','UNAUTHORIZED') NOT NULL,
    entry_time   DATETIME     NOT NULL,
    exit_time    DATETIME,
    duration_s   FLOAT,
    snapshot     VARCHAR(512),
    camera_id    VARCHAR(64),
    camera_label VARCHAR(128),
    location     VARCHAR(128),
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL,
    INDEX idx_entry_time (entry_time),
    INDEX idx_status     (status),
    INDEX idx_person_id  (person_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Snapshots table: metadata for every saved image
CREATE TABLE IF NOT EXISTS snapshots (
    id          BIGINT      AUTO_INCREMENT PRIMARY KEY,
    filename    VARCHAR(256) NOT NULL,
    filepath    VARCHAR(512) NOT NULL,
    person_name VARCHAR(128),
    camera_id   VARCHAR(64),
    event_id    BIGINT,
    captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id)   REFERENCES access_events(id) ON DELETE SET NULL,
    FOREIGN KEY (camera_id)  REFERENCES cameras(id)       ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id         BIGINT      AUTO_INCREMENT PRIMARY KEY,
    level      VARCHAR(16) NOT NULL,   -- INFO, WARNING, ERROR
    module     VARCHAR(64),
    message    TEXT        NOT NULL,
    logged_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_level    (level),
    INDEX idx_logged_at (logged_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def init_schema():
    """Create all tables if they don't exist. Safe to call multiple times."""
    conn = get_connection()
    cur  = conn.cursor()
    for statement in SCHEMA_SQL.strip().split(';'):
        stmt = statement.strip()
        if stmt:
            cur.execute(stmt)
    conn.commit()
    cur.close()
    conn.close()
    print('[DB] Schema initialised (all tables ready).')


# ── Utility helpers ────────────────────────────────────────────────────────

def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _serialize(rows):
    """
    Convert any datetime/date values in a list of dicts to plain strings.
    MySQL returns datetime columns as Python datetime objects; Jinja2 templates
    (written for SQLite plain strings) crash when they receive them.
    Call this on every function that returns rows to templates.
    """
    fmt = '%Y-%m-%d %H:%M:%S'
    result = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if hasattr(v, 'strftime'):
                clean[k] = v.strftime(fmt)
            else:
                clean[k] = v
        result.append(clean)
    return result


# ── Camera helpers ─────────────────────────────────────────────────────────

def upsert_camera(cam_id, label, source='webcam', rtsp_url='',
                  location_name='', floor='', building='',
                  location_lat=None, location_lng=None,
                  reconnect_attempts=5):
    """Insert or update a camera row."""
    sql = """
        INSERT INTO cameras
            (id, label, source, rtsp_url, location_name, floor, building,
             location_lat, location_lng, reconnect_attempts)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            label=VALUES(label), source=VALUES(source),
            rtsp_url=VALUES(rtsp_url), location_name=VALUES(location_name),
            floor=VALUES(floor), building=VALUES(building),
            location_lat=VALUES(location_lat), location_lng=VALUES(location_lng),
            reconnect_attempts=VALUES(reconnect_attempts)
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(sql, (cam_id, label, source, rtsp_url, location_name,
                      floor, building, location_lat, location_lng, reconnect_attempts))
    conn.commit()
    cur.close()
    conn.close()


def get_all_cameras():
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute('SELECT * FROM cameras ORDER BY added_at')
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return _serialize(rows)


def remove_camera(cam_id):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute('UPDATE cameras SET active=0 WHERE id=%s', (cam_id,))
    conn.commit()
    cur.close()
    conn.close()


# ── Person helpers ─────────────────────────────────────────────────────────

def upsert_person(person_id, name, role='', contact='',
                  department='', access_level='standard', notes=''):
    sql = """
        INSERT INTO persons (id, name, role, contact, department, access_level, notes)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name), role=VALUES(role), contact=VALUES(contact),
            department=VALUES(department), access_level=VALUES(access_level),
            notes=VALUES(notes)
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(sql, (person_id, name, role, contact, department, access_level, notes))
    conn.commit()
    cur.close()
    conn.close()


def get_all_persons():
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute('SELECT * FROM persons WHERE active=1 ORDER BY name')
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return _serialize(rows)


def deactivate_person(person_id):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute('UPDATE persons SET active=0 WHERE id=%s', (person_id,))
    conn.commit()
    cur.close()
    conn.close()


# ── Access event helpers ───────────────────────────────────────────────────

def record_entry(person_name, person_id=None, status='AUTHORIZED',
                 snapshot=None, camera_id=None, camera_label='webcam',
                 location=None):
    """Insert entry row. Returns the new row id."""
    sql = """
        INSERT INTO access_events
            (person_name, person_id, status, entry_time, snapshot,
             camera_id, camera_label, location)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(sql, (person_name, person_id, status, _now(),
                      snapshot, camera_id, camera_label, location))
    conn.commit()
    row_id = cur.lastrowid
    cur.close()
    conn.close()
    return row_id


def record_exit(row_id, exit_time=None):
    """Update row with exit time and duration."""
    if not row_id:
        return
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute('SELECT entry_time FROM access_events WHERE id=%s', (row_id,))
    row = cur.fetchone()
    if row:
        entry_dt = row[0]  # MySQL returns a datetime object directly
        if not hasattr(entry_dt, 'strftime'):
            # Fallback: parse string if somehow stored as text
            entry_dt = datetime.strptime(str(entry_dt), '%Y-%m-%d %H:%M:%S')
        exit_dt  = datetime.now()
        duration = (exit_dt - entry_dt).total_seconds()
        cur.execute(
            'UPDATE access_events SET exit_time=%s, duration_s=%s WHERE id=%s',
            (exit_dt.strftime('%Y-%m-%d %H:%M:%S'), duration, row_id)
        )
        conn.commit()
    cur.close()
    conn.close()


def get_recent_events(limit=100):
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, person_name, person_id, status, entry_time, exit_time,
               duration_s, snapshot, camera_id, camera_label, location
        FROM access_events
        ORDER BY id DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return _serialize(rows)


def get_stats():
    conn  = get_connection()
    cur   = conn.cursor(dictionary=True)
    today = datetime.now().strftime('%Y-%m-%d')

    cur.execute('SELECT COUNT(*) AS cnt FROM access_events')
    total_events = cur.fetchone()['cnt']

    cur.execute("""SELECT COUNT(*) AS cnt FROM access_events
                   WHERE status='AUTHORIZED' AND DATE(entry_time)=%s""", (today,))
    today_auth = cur.fetchone()['cnt']

    cur.execute("""SELECT COUNT(*) AS cnt FROM access_events
                   WHERE status='UNAUTHORIZED' AND DATE(entry_time)=%s""", (today,))
    today_unauth = cur.fetchone()['cnt']

    cur.execute('SELECT COUNT(*) AS cnt FROM persons WHERE active=1')
    total_persons = cur.fetchone()['cnt']

    cur.close()
    conn.close()

    return {
        'total_events':       total_events,
        'today_authorized':   today_auth,
        'today_unauthorized': today_unauth,
        'total_persons':      total_persons,
    }


# ── Snapshot helpers ───────────────────────────────────────────────────────

def record_snapshot(filename, filepath, person_name=None,
                    camera_id=None, event_id=None):
    sql = """INSERT INTO snapshots
             (filename, filepath, person_name, camera_id, event_id)
             VALUES (%s,%s,%s,%s,%s)"""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(sql, (filename, filepath, person_name, camera_id, event_id))
    conn.commit()
    snap_id = cur.lastrowid
    cur.close()
    conn.close()
    return snap_id


def get_all_snapshots(limit=200):
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute('SELECT * FROM snapshots ORDER BY captured_at DESC LIMIT %s', (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return _serialize(rows)


# ── System log helper ──────────────────────────────────────────────────────

def log_system(level, message, module='system'):
    try:
        sql = 'INSERT INTO system_logs (level, module, message) VALUES (%s,%s,%s)'
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (level.upper(), module, message))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f'[DB] system_log insert failed: {e}')


# ── Quick self-test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('[DB] Running self-test...')
    init_schema()
    print('[DB] Inserting test camera...')
    upsert_camera('cam_test', 'Test Camera', location_name='Entrance', floor='Ground')
    print('[DB] Cameras:', get_all_cameras())
    print('[DB] Self-test passed.')
