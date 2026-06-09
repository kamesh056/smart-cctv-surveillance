# 🛡️ NeuralGuard — Smart CCTV Surveillance System

> **AI-powered security monitoring with real-time face recognition, YOLOv8 person detection, hazard detection, camera tamper alerts, and a full web management portal.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [How It Works](#how-it-works)
- [Detection Pipeline](#detection-pipeline)
- [Web Portal](#web-portal)
- [Camera Setup](#camera-setup)
- [Database Setup](#database-setup)
- [Troubleshooting](#troubleshooting)
- [Known Issues & Fixes](#known-issues--fixes)

---

## Overview

**NeuralGuard** is a production-grade, Python-based CCTV surveillance system built for large-scale industrial and enterprise environments. It replaces passive, human-monitored camera feeds with an intelligent, always-on detection platform that:

- Automatically **identifies authorised vs unauthorised individuals** in real time
- Runs **YOLOv8 person detection** as a first-stage filter before face recognition
- Detects **workplace hazards** (fire, flood) via a separate YOLO-based hazard model
- Alerts on **camera blockage or tampering** using multi-metric image analysis
- Triggers a **hardware alarm** (buzzer via Arduino/serial) on intrusion
- Maintains a **complete, searchable audit log** with timestamped snapshots
- Provides a **browser-based management portal** for enrolment, monitoring, and log review

---

## Features

### 🔍 Detection & Recognition
| Feature | Detail |
|---|---|
| Face Recognition | dlib HOG/CNN models via `face_recognition` library |
| Person Detection | YOLOv8n (`yolov8n.pt`) — runs every 6th frame |
| Hazard Detection | YOLOv8n-fire (`yolov8n-fire.pt`) — fire + flood detection |
| Camera Tamper | Brightness, variance, and edge-density based blockage detection |
| Two-Gap Confidence Check | Rejects ambiguous matches (gap < 0.05) → labelled UNKNOWN |
| Facial Landmark Validation | Eyes + nose + lips must be present — eliminates false triggers |
| Min Face Size Filter | 4% of frame height minimum — ignores distant noise |

### 🔔 Alerting
- Hardware buzzer via **PySerial** (Arduino/serial port)
- Email notifications via SMTP (configurable)
- Real-time portal dashboard with annotated **MJPEG live feed**
- Per-camera **alert cooldown** (default: 120 seconds)

### 🌐 Web Portal
- Secure login with **CSRF protection** and 7-day remember-me sessions
- Person enrolment via **live webcam recording** or **video file upload**
- Real-time **access log** with date/identity/status filtering
- **Intrusion snapshot** gallery
- Multi-camera management (USB webcam, RTSP IP camera, HTTP stream)
- Live status bar: `Persons: N | Auth: N | Unknown: N`

### 🗄️ Data & Persistence
- **MySQL** production database (pool size: 10)
- SQLite supported for development/testing
- Migration utility: `migrate_sqlite_to_mysql.py`
- Face encodings stored in `face_encodings.pkl` with hot-reload support

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         startup.py                              │
│              (Single launcher — cross-platform)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ spawns
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         portal.py                               │
│              (Flask app — NeuralGuard v15)                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ camera_stream│  │ yolo_detector│  │  hazard_detector     │  │
│  │   .py        │  │   .py        │  │     .py              │  │
│  │ (USB/RTSP/   │  │ (YOLOv8n     │  │ (fire model +        │  │
│  │  HTTP)       │  │  every 6th   │  │  flood HSV check)    │  │
│  └──────┬───────┘  │  frame)      │  └──────────────────────┘  │
│         │          └──────────────┘                             │
│         │ frames (pickle IPC)                                   │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐              │
│  │          recog_worker.py (subprocess)         │              │
│  │  • Length-prefixed binary IPC (stdin/stdout) │              │
│  │  • 0xFFFFFFFF RELOAD sentinel                │              │
│  │  • face_encodings.pkl mtime auto-reload      │              │
│  └──────────────┬───────────────────────────────┘              │
│                 │ results                                       │
│                 ▼                                               │
│  ┌──────────────────────────────────────────────┐              │
│  │         face_recognizer.py (v5)              │              │
│  │  • Tolerance: 0.50                           │              │
│  │  • Two-gap check: 0.05                       │              │
│  │  • Landmark validation (_is_real_face)       │              │
│  │  • Min face height ratio: 0.04               │              │
│  └──────┬───────────────────────────────────────┘              │
│         │                                                       │
│    ┌────┴────────────────────┐                                  │
│    │ AUTHORISED              │ UNKNOWN / UNAUTHORISED           │
│    │ Green box + log entry   │ Red box + alert_system.py        │
│    └─────────────────────────┤   + snapshot_manager.py         │
│                              │   + access_logger.py            │
│                              └──────────────────────────────── │
│                                                                 │
│  blockage_wiring.py ─── wired to all active camera streams     │
│  (brightness / variance / edge-density tamper detection)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Face Recognition | `face-recognition` (dlib HOG/CNN), OpenCV 4.8.1.78 |
| Person Detection | YOLOv8n (`ultralytics`) — `yolov8n.pt` |
| Hazard Detection | YOLOv8n-fire (`yolov8n-fire.pt`) + HSV flood detection |
| Web Framework | Flask + gevent WSGI |
| Frontend | HTML5, CSS3, JavaScript; Jinja2 templates; MJPEG stream |
| Database | MySQL (`mysql-connector-python`), SQLite (dev) |
| Hardware Alarm | PySerial — Arduino/buzzer over serial port |
| Image Processing | OpenCV, Pillow, NumPy |
| Config | `config.json` + `python-dotenv` |
| IPC | Length-prefixed binary pickle over stdin/stdout pipes |

---

## Project Structure

```
smart-cctv-surveillance/
│
├── portal.py                  # Main Flask app (NeuralGuard v15, ~2200 lines)
├── startup.py                 # Single-click launcher (cross-platform)
├── face_recognizer.py         # Face recognition engine (v5)
├── recog_worker.py            # Recognition subprocess + IPC handler (v3)
├── camera_stream.py           # Thread-safe camera capture (USB/RTSP/HTTP)
├── yolo_detector.py           # YOLOv8 person detection singleton
├── hazard_detector.py         # Fire/flood hazard detection
├── blockage_wiring.py         # Camera tamper/blockage detection
├── alert_system.py            # Hardware buzzer + DB event logging
├── snapshot_manager.py        # Timestamped intrusion image archival
├── access_logger.py           # MySQL audit trail writer
├── db_mysql.py                # MySQL connection pool + CRUD helpers
├── video_face_encoder.py      # Enrolment video → face_encodings.pkl
├── migrate_sqlite_to_mysql.py # Dev→production DB migration utility
│
├── config.json                # All runtime configuration
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── authorized_persons/        # Enrolment video/image storage
├── static/images/             # Static assets
├── templates/                 # Jinja2 HTML templates
├── logs/                      # portal.log, recog_worker.log
│
└── MYSQL_MIGRATION_GUIDE.md   # Step-by-step MySQL setup guide
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- MySQL 8.0 (for production) or SQLite (for development)
- A C++ compiler (required by `dlib` — see note below)
- `cmake` installed and on PATH
- Git

### 1. Clone the repository

```bash
git clone https://github.com/kamesh056/smart-cctv-surveillance.git
cd smart-cctv-surveillance
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on `dlib`:** Installing `dlib` requires CMake and a C++ build toolchain.
> - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and [CMake](https://cmake.org/download/)
> - **Ubuntu/Debian:** `sudo apt-get install build-essential cmake`
> - **macOS:** `xcode-select --install && brew install cmake`

### 4. Download YOLO model weights

```bash
# YOLOv8 nano (person detection)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# YOLOv8 fire model (hazard detection) — place in project root
# Download from: https://github.com/ultralytics/assets or your preferred source
# Expected filename: yolov8n-fire.pt
```

### 5. Set up the database

Follow the instructions in [`MYSQL_MIGRATION_GUIDE.md`](MYSQL_MIGRATION_GUIDE.md), or for quick SQLite development setup, no action is needed — the system auto-creates tables on first run.

---

## Configuration

All settings live in **`config.json`**. Key sections:

### Camera Setup

```json
"cameras": [
  {
    "id": "cam_main",
    "label": "Main Entrance",
    "source": "rtsp",
    "rtsp_url": "rtsp://user:password@192.168.1.100:554/Streaming/Channels/101",
    "reconnect_attempts": 5,
    "hazard_enabled": true
  },
  {
    "id": "cam_usb",
    "label": "Reception",
    "source": "webcam",
    "rtsp_url": "",
    "reconnect_attempts": 5,
    "hazard_enabled": false
  }
]
```

`source` can be `"rtsp"`, `"webcam"`, or `"http"`.

### Face Recognition Tuning

```json
"recognition": {
  "tolerance": 0.5,          // Lower = stricter matching (range: 0.4–0.6)
  "model": "hog",            // "hog" (faster) or "cnn" (more accurate, needs GPU)
  "upsample": 1,             // 0=normal range, 1–2=extended range detection
  "scale_factor": 0.5,       // Frame scale before processing (0.5–1.0)
  "min_face_height_ratio": 0.04,  // Min face size as fraction of frame height
  "min_distance_gap": 0.05   // Two-gap confidence threshold
}
```

### YOLO Person Detection

```json
"yolo": {
  "enabled": true,
  "model": "yolov8n.pt",     // Path to YOLO weights file
  "confidence": 0.4,         // Detection confidence threshold
  "classes": [0],            // 0 = person class in COCO
  "process_every": 6,        // Run YOLO every N frames
  "use_openvino": true       // Use OpenVINO backend if available (faster on Intel)
}
```

### Hazard Detection

```json
"hazard": {
  "enabled": true,
  "fire_model": "yolov8n-fire.pt",
  "confidence": 0.55,
  "process_every": 15,       // Run hazard detection every 15 frames
  "cooldown_seconds": 120
}
```

### Camera Blockage Detection

```json
"blockage_detection": {
  "brightness_thresh": 15.0,    // Average brightness below this = dark/blocked
  "variance_thresh": 20.0,      // Low variance = uniform (blocked) frame
  "edge_density_thresh": 0.002, // Very few edges = covered lens
  "confirm_seconds": 3.0,       // Sustained blockage duration before alert
  "cooldown_seconds": 120.0,
  "alert_on_clear": true        // Also alert when blockage is lifted
}
```

### Portal & Admin

```json
"portal": {
  "host": "0.0.0.0",
  "port": 5000,
  "secret_key": "change-this-to-a-strong-random-key",
  "admin_email": "admin@yourorg.com",
  "admin_password": "change-this-password"
}
```

> ⚠️ **Security:** Always change `secret_key` and `admin_password` before deploying. Never commit credentials to version control.

### MySQL

```json
"mysql": {
  "host": "localhost",
  "port": 3306,
  "user": "neuralguard",
  "password": "your_db_password",
  "database": "neuralguard",
  "pool_size": 10
}
```

---

## Running the System

### Quick Start (recommended)

```bash
python startup.py
```

This will:
1. Stop any existing portal instance
2. Launch `portal.py` as a background process
3. Wait up to 15 seconds for the portal to be ready
4. Automatically open `http://127.0.0.1:5000` in your browser

### Stop the system

```bash
python startup.py --stop
```

### Manual start (for development/debugging)

```bash
python portal.py
```

Logs are written to `logs/portal.log` and `logs/recog_worker.log`.

### Default Login

| Field | Value |
|---|---|
| Email | `admin@jindalpower.com` *(change in config.json)* |
| Password | `admin@12345` *(change in config.json)* |

---

## How It Works

### Detection Flow (per frame)

```
Camera Frame
    │
    ├── Every 6th frame: YOLOv8 person detection
    │       → Identifies person bounding boxes in the full frame
    │
    ├── Every 15th frame: Hazard detection (fire model + HSV flood check)
    │
    ├── recog_worker.py receives frame via binary IPC pipe
    │       → face_recognizer.py processes detected regions:
    │           1. Size filter (min_face_height_ratio = 0.04)
    │           2. Facial landmark validation (eyes, nose, lips)
    │           3. Face encoding (128-dimension vector)
    │           4. Euclidean distance matching vs enrolled persons
    │           5. Two-gap confidence check (gap < min_distance_gap → UNKNOWN)
    │
    ├── AUTHORISED person detected
    │       → Green bounding box drawn
    │       → access_logger.py writes entry to MySQL
    │
    └── UNKNOWN / UNAUTHORISED person detected
            → Red bounding box drawn
            → alert_system.py fires buzzer (PySerial) + respects cooldown
            → snapshot_manager.py saves timestamped JPEG
            → access_logger.py writes intrusion event to MySQL
            → Portal dashboard updates in real time
```

### Person Enrolment Flow

```
Portal → Upload video or record via webcam (MediaRecorder API)
    → video_face_encoder.py extracts face frames from video
    → Generates 128-dim encoding per face
    → Saves to face_encodings.pkl
    → Portal sets _recognizer = None  (forces hot-reload)
    → recog_worker.py receives RELOAD sentinel (0xFFFFFFFF)
    → Reloads face_encodings.pkl from disk
    → New person recognised on the very next frame
```

---

## Web Portal

Access at: **`http://127.0.0.1:5000`** (or your server IP if deployed on a network)

### Pages

| Page | Description |
|---|---|
| **Dashboard** | Live MJPEG camera feeds with annotated bounding boxes and status bar |
| **Persons** | Manage authorised persons — add, view, deactivate |
| **Enrol** | Record a short video via webcam or upload a video file to enrol a person |
| **Access Logs** | Searchable, filterable audit trail of all detection events |
| **Snapshots** | Gallery of all captured intrusion images |
| **Cameras** | Add, configure, and manage camera sources |
| **Alerts** | View blockage and hazard alert history |
| **Settings** | Update portal configuration |

### Status Bar

The live feed status bar shows real-time detection state:

```
● LIVE  |  Persons: 2  |  Auth: 1  |  Unknown: 1  |  14:32:07
```

---

## Camera Setup

### USB Webcam

```json
{ "source": "webcam", "rtsp_url": "" }
```

### RTSP IP Camera (e.g. Hikvision)

```json
{
  "source": "rtsp",
  "rtsp_url": "rtsp://username:password@192.168.1.x:554/Streaming/Channels/101"
}
```

The system uses `RtspFrameBuffer` — a dedicated background thread that continuously drains the RTSP buffer via `cap.grab()`, eliminating the 15–20 second lag common in naive RTSP implementations.

### HTTP Stream

```json
{ "source": "http", "rtsp_url": "http://192.168.1.x:8080/video" }
```

---

## Database Setup

For a fresh MySQL installation:

```sql
CREATE DATABASE neuralguard CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'neuralguard'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON neuralguard.* TO 'neuralguard'@'localhost';
FLUSH PRIVILEGES;
```

Update `config.json` with your credentials, then run the system — tables are created automatically on first launch.

For migration from an existing SQLite database, see [`MYSQL_MIGRATION_GUIDE.md`](MYSQL_MIGRATION_GUIDE.md):

```bash
python migrate_sqlite_to_mysql.py
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Portal doesn't start | Check `logs/portal.log` for errors |
| Face not recognised | Lower `tolerance` (try 0.45); ensure good lighting during enrolment video |
| Too many false alarms | Raise `tolerance` slightly; check `min_distance_gap` |
| RTSP feed lagging | Already handled by `RtspFrameBuffer`; check network bandwidth to camera |
| `dlib` install fails | Ensure CMake and C++ build tools are installed (see Installation section) |
| YOLO model not found | Run the YOLO download command in the Installation section |
| MySQL connection error | Verify credentials in `config.json` match your MySQL user |
| Buzzer not firing | Check serial port in `alert_system.py`; ensure Arduino is connected |
| New person not recognised | Portal auto-sends RELOAD signal; check `logs/recog_worker.log` |
| 100% CPU usage | Reduce `process_every` for YOLO, or lower camera resolution in `config.json` |

---

## Known Issues & Fixes

These are documented fixes applied in **portal.py v15** and **recog_worker.py v3**:

| Fix | Description |
|---|---|
| **FIX 1** — gevent LoopExit | `_GLOBAL_RECOG_Q` created before `monkey.patch_all(thread=False, subprocess=False)` to preserve stdlib queue |
| **FIX 2/3** — Recognizer not reloading | After enrolment, `_recognizer = None` forces a full reload on the next frame |
| **FIX 4** — IPC RELOAD sentinel | `0xFFFFFFFF` length header signals worker to reload without restarting the process |
| **FIX 5** — Worker crash recovery | Outer retry loop in portal restarts `recog_worker.py` subprocess after 5-second cooldown |
| **FIX 6** — Per-camera lock contention | Replaced global OpenCV lock with `_get_cam_lock(camera_id)` factory — cameras read concurrently |
| **FIX 7** — 100% CPU spin | Added 33ms minimum sleep between webcam reads (~30 fps cap) |
| **FIX 9** — Unnecessary YOLO IPC | Skips full-frame YOLO subprocess call when all visible persons are already confirmed authorised |
| **AUTO-RELOAD** | `recog_worker.py` checks `face_encodings.pkl` mtime on every frame — picks up changes within one frame |

---

## License

This project was developed as part of an industrial internship at **Jindal Power Limited, IT Department, Tamnar, Chhattisgarh**. All rights reserved.

---

*Built with Python, OpenCV, dlib, YOLOv8, Flask, and MySQL.*
