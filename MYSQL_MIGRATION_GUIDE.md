# NeuralGuard — MySQL Integration Guide

## What Changes

| Before | After |
|---|---|
| SQLite file `logs/access_events.db` | MySQL database `neuralguard` |
| Single table `access_events` | 5 tables: cameras, persons, access_events, snapshots, system_logs |
| Camera info only in `config.json` | Camera info in MySQL with location, floor, building, GPS |
| No person records in DB | Persons table mirrors `authorized_persons/` folder |
| No snapshot tracking in DB | Snapshots table records every saved image |

---

## New Files

| File | Purpose |
|---|---|
| `db_mysql.py` | Central MySQL module — all DB logic lives here |
| `portal.py` | Updated — uses `db_mysql` instead of SQLite |
| `access_logger.py` | Updated — uses `db_mysql` |
| `config.json` | Updated — added `mysql` section and camera location fields |
| `requirements.txt` | Updated — added `mysql-connector-python` |

---

## Step 1 — Install MySQL Server

### On Windows
1. Download MySQL Community Server from https://dev.mysql.com/downloads/mysql/
2. Run the installer, choose **Developer Default**
3. Set a root password when prompted — write it down
4. MySQL will start automatically as a Windows service

### On Ubuntu / Debian (Linux)
```bash
sudo apt update
sudo apt install mysql-server -y
sudo systemctl start mysql
sudo systemctl enable mysql
sudo mysql_secure_installation   # follow prompts, set root password
```

### On macOS
```bash
brew install mysql
brew services start mysql
mysql_secure_installation
```

---

## Step 2 — Create the Database and User

Open a MySQL shell (replace `YourRootPassword` with your actual root password):

```bash
mysql -u root -p
```

Once inside MySQL, run these commands **exactly**:

```sql
-- Create the database
CREATE DATABASE neuralguard CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create a dedicated user (do NOT use root for the app)
CREATE USER 'neuralguard'@'localhost' IDENTIFIED BY 'StrongPassword123!';

-- Give the user full access to only this database
GRANT ALL PRIVILEGES ON neuralguard.* TO 'neuralguard'@'localhost';

-- Apply changes
FLUSH PRIVILEGES;

-- Verify
SHOW DATABASES;

-- Exit
EXIT;
```

> **Security tip:** Change `StrongPassword123!` to something unique. Never use root credentials in application config.

---

## Step 3 — Configure Credentials

You have two options. Choose ONE.

### Option A — Edit config.json (simplest)

Open `config.json` and find the `mysql` section:

```json
"mysql": {
  "host": "localhost",
  "port": 3306,
  "user": "neuralguard",
  "password": "StrongPassword123!",
  "database": "neuralguard"
}
```

Replace `StrongPassword123!` with the password you chose in Step 2.

### Option B — Use a .env file (more secure, keeps secrets out of config)

Create a file named `.env` in your project folder:

```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=neuralguard
MYSQL_PASSWORD=StrongPassword123!
MYSQL_DATABASE=neuralguard
```

The app reads `.env` automatically via `python-dotenv`. This is the recommended approach if you use Git — add `.env` to your `.gitignore`.

---

## Step 4 — Install the New Python Package

```bash
pip install mysql-connector-python
```

Or reinstall everything from the updated requirements file:

```bash
pip install -r requirements.txt
```

---

## Step 5 — Replace the Project Files

Copy the updated files into your project folder, overwriting the originals:

```
your_project/
├── db_mysql.py          ← NEW (add this file)
├── portal.py            ← REPLACE
├── access_logger.py     ← REPLACE
├── config.json          ← REPLACE (update password in mysql section)
├── requirements.txt     ← REPLACE
└── ... (all other files stay the same)
```

The files you do NOT need to change:
- `face_recognizer.py`
- `camera_stream.py`
- `alert_system.py`
- `snapshot_manager.py`
- `startup.py`
- `video_face_encoder.py`

---

## Step 6 — Test the Database Connection

Run the self-test built into `db_mysql.py`:

```bash
python db_mysql.py
```

Expected output:
```
[DB] Connected to MySQL at localhost:3306 — database 'neuralguard'
[DB] Schema initialised (all tables ready)
[DB] Inserting test camera...
[DB] Cameras: [{'id': 'cam_test', 'label': 'Test Camera', ...}]
[DB] Self-test passed.
```

If you see an error like `Access denied` or `Can't connect`, recheck your password in Step 2 and Step 3.

---

## Step 7 — Run the Application

Launch normally:

```bash
python startup.py
```

or directly:

```bash
python portal.py
```

On first startup, `portal.py` will automatically call `db.init_schema()` which creates all 5 tables in MySQL. You will see:

```
[DB] Schema initialised (all tables ready)
[PORTAL] Starting — http://0.0.0.0:5000
```

---

## Step 8 — Add Camera Location Details

When adding a camera through the portal at `http://localhost:5000/cameras`, you now have extra fields:

| Field | Example | Stored in |
|---|---|---|
| Label | "Main Gate Camera" | MySQL `cameras` table |
| Source | webcam / RTSP | MySQL `cameras` table |
| Location Name | "Main Entrance" | MySQL `cameras` table |
| Floor | "Ground Floor" | MySQL `cameras` table |
| Building | "Block A" | MySQL `cameras` table |
| GPS Latitude | 22.5726 | MySQL `cameras` table |
| GPS Longitude | 88.3639 | MySQL `cameras` table |

You can also add location details directly to `config.json` under each camera entry:

```json
"cameras": [
  {
    "id": "cam1780651075",
    "label": "Main Camera",
    "source": "webcam",
    "rtsp_url": "",
    "reconnect_attempts": 5,
    "location_name": "Main Entrance",
    "floor": "Ground Floor",
    "building": "Block A",
    "location_lat": 22.5726,
    "location_lng": 88.3639
  }
]
```

These are synced to MySQL automatically when the detection thread starts.

---

## MySQL Table Reference

### `cameras` — All registered cameras

| Column | Type | Description |
|---|---|---|
| id | VARCHAR(64) | Unique camera ID (e.g. `cam1780651075`) |
| label | VARCHAR(128) | Display name |
| source | VARCHAR(16) | `webcam` or `rtsp` |
| rtsp_url | VARCHAR(512) | RTSP stream URL if applicable |
| location_name | VARCHAR(128) | Human readable location |
| floor | VARCHAR(32) | Floor number/name |
| building | VARCHAR(64) | Building name |
| location_lat | DECIMAL(10,7) | GPS latitude |
| location_lng | DECIMAL(10,7) | GPS longitude |
| reconnect_attempts | INT | Max reconnect tries |
| active | TINYINT | 1=active, 0=removed |
| added_at | DATETIME | When first added |

### `persons` — Authorised enrolled people

| Column | Type | Description |
|---|---|---|
| id | VARCHAR(128) | Folder name (e.g. `ayush_sharma_1234567890`) |
| name | VARCHAR(128) | Full display name |
| role | VARCHAR(64) | Job title |
| contact | VARCHAR(128) | Phone or email |
| department | VARCHAR(64) | Department |
| access_level | VARCHAR(32) | `standard`, `admin`, etc. |
| enrolled_at | DATETIME | Enrollment timestamp |
| active | TINYINT | 1=active, 0=removed |

### `access_events` — Entry/exit log

| Column | Type | Description |
|---|---|---|
| id | BIGINT | Auto-increment primary key |
| person_name | VARCHAR(128) | Name or "UNKNOWN" |
| person_id | VARCHAR(128) | Matches `persons.id` |
| status | ENUM | `AUTHORIZED` or `UNAUTHORIZED` |
| entry_time | DATETIME | When first detected |
| exit_time | DATETIME | When left the frame |
| duration_s | FLOAT | Time in seconds |
| snapshot | VARCHAR(512) | Path to snapshot file |
| camera_id | VARCHAR(64) | Which camera |
| camera_label | VARCHAR(128) | Camera display name |
| location | VARCHAR(128) | Camera location name |

### `snapshots` — Saved image references

| Column | Type | Description |
|---|---|---|
| id | BIGINT | Auto-increment primary key |
| filename | VARCHAR(256) | File name only |
| filepath | VARCHAR(512) | Full path on disk |
| person_name | VARCHAR(128) | Who was detected |
| camera_id | VARCHAR(64) | Which camera |
| event_id | BIGINT | Linked access_events row |
| captured_at | DATETIME | When saved |

### `system_logs` — Application messages

| Column | Type | Description |
|---|---|---|
| id | BIGINT | Auto-increment primary key |
| level | VARCHAR(16) | INFO / WARNING / ERROR |
| module | VARCHAR(64) | Which module logged it |
| message | TEXT | Log message |
| logged_at | DATETIME | Timestamp |

---

## Useful MySQL Queries

Open MySQL shell and run these to inspect your data:

```sql
USE neuralguard;

-- All cameras with location
SELECT id, label, location_name, floor, building FROM cameras;

-- Today's access events
SELECT person_name, status, entry_time, exit_time, location
FROM access_events
WHERE DATE(entry_time) = CURDATE()
ORDER BY entry_time DESC;

-- Count of unauthorized entries per day (last 7 days)
SELECT DATE(entry_time) AS day, COUNT(*) AS intrusions
FROM access_events
WHERE status = 'UNAUTHORIZED'
  AND entry_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY day
ORDER BY day;

-- All snapshots for a specific camera
SELECT s.filename, s.captured_at, s.person_name
FROM snapshots s
JOIN cameras c ON s.camera_id = c.id
WHERE c.label = 'Main Camera'
ORDER BY s.captured_at DESC;

-- Most active persons today
SELECT person_name, COUNT(*) AS visits
FROM access_events
WHERE DATE(entry_time) = CURDATE() AND status = 'AUTHORIZED'
GROUP BY person_name
ORDER BY visits DESC;

-- Recent system errors
SELECT module, message, logged_at
FROM system_logs
WHERE level = 'ERROR'
ORDER BY logged_at DESC
LIMIT 20;
```

---

## Troubleshooting

### `mysql.connector.errors.DatabaseError: 1045 Access denied`
Your password in `config.json` or `.env` does not match what you set in MySQL.
Re-run Step 2 to reset the password:
```sql
ALTER USER 'neuralguard'@'localhost' IDENTIFIED BY 'NewPassword';
FLUSH PRIVILEGES;
```
Then update `config.json` to match.

### `mysql.connector.errors.InterfaceError: 2003 Can't connect`
MySQL is not running. Start it:
- **Windows**: Open Services → find `MySQL80` → Start
- **Linux**: `sudo systemctl start mysql`
- **Mac**: `brew services start mysql`

### `ModuleNotFoundError: No module named 'mysql'`
```bash
pip install mysql-connector-python
```

### Portal starts but shows no events
If you had data in the old SQLite database and want to keep it, run this migration script once:

```python
# migrate_sqlite_to_mysql.py  — run ONCE to copy old data
import sqlite3, db_mysql

db_mysql.init_schema()
old = sqlite3.connect('logs/access_events.db')
old.row_factory = sqlite3.Row
rows = old.execute('SELECT * FROM access_events ORDER BY id').fetchall()
conn = db_mysql.get_connection()
cur  = conn.cursor()
for r in rows:
    cur.execute("""
        INSERT IGNORE INTO access_events
        (person_name, person_id, status, entry_time, exit_time, duration_s,
         snapshot, camera_label, location)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (r['person_name'], r['person_id'], r['status'],
          r['entry_time'], r['exit_time'], r.get('duration_s'),
          r['snapshot'], r['camera'], None))
conn.commit()
cur.close()
conn.close()
old.close()
print(f'Migrated {len(rows)} rows.')
```

Run with: `python migrate_sqlite_to_mysql.py`

---

## Remote MySQL (Optional)

If MySQL runs on a different machine or server:

1. On the MySQL server, create the user with the remote IP:
```sql
CREATE USER 'neuralguard'@'192.168.1.100' IDENTIFIED BY 'StrongPassword123!';
GRANT ALL PRIVILEGES ON neuralguard.* TO 'neuralguard'@'192.168.1.100';
FLUSH PRIVILEGES;
```

2. In `config.json`, set `"host"` to the MySQL server's IP:
```json
"mysql": {
  "host": "192.168.1.50",
  "port": 3306,
  ...
}
```

3. Make sure port 3306 is open in the server's firewall.

---

## Summary of Changes by File

| File | What changed |
|---|---|
| `db_mysql.py` | NEW — all MySQL logic: connection pool, schema, CRUD helpers |
| `portal.py` | Replaced `sqlite3` with `db_mysql`; camera forms now include location fields; persons synced to MySQL on enroll |
| `access_logger.py` | Replaced `sqlite3` with `db_mysql`; same public API (`record_entry`, `record_exit`) |
| `config.json` | Added `mysql` block; camera entries now support `location_name`, `floor`, `building`, `location_lat`, `location_lng` |
| `requirements.txt` | Added `mysql-connector-python` |
