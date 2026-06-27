# 🚦 Smart Traffic & Emergency Vehicle Detection System

> **An AI-powered real-time traffic monitoring system that detects vehicles, tracks them across frames, estimates their speed, adapts traffic signals automatically, and fires instant SOS alerts to Telegram the moment an emergency vehicle appears.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![DeepSort](https://img.shields.io/badge/DeepSort-Realtime-orange)
![Telegram](https://img.shields.io/badge/Telegram-Bot%20API-26A5E4?logo=telegram)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What Is This Project?

The **Smart Traffic & Emergency Vehicle Detection System** is a computer vision pipeline built on top of **YOLOv8** and **DeepSort** that processes traffic video in real time. It does five things simultaneously:

1. **Detects** every vehicle in the scene (cars, trucks, buses, motorcycles)
2. **Tracks** each vehicle persistently frame-to-frame so it always has a unique ID
3. **Estimates** the speed of each tracked vehicle in km/h
4. **Adapts** the traffic signal green/red timing based on live vehicle density
5. **Alerts** operators on Telegram the instant an emergency vehicle (truck/bus) is spotted, attaching a photo snapshot of that exact frame

It supports both **static image analysis** (single image → annotated result + confidence chart) and **live video / webcam analysis** (full tracking + analytics pipeline).

---

## 🚨 What Problem Does It Solve?

### The Problem: Manual, Reactive Traffic Management

Traditional traffic systems are **timer-based and blind**:
- Signals run fixed green/red cycles regardless of how many vehicles are waiting
- No one is watching for emergency vehicles — an ambulance or fire truck stuck at a red light costs lives
- Traffic managers cannot monitor every intersection simultaneously
- Incident response is reactive: someone must notice a problem and call it in

This causes:
- ⏱ Unnecessary delays for emergency vehicles
- 🚗 Traffic congestion from poorly timed signals
- 👁 No real-time visibility into what is actually on the road
- 📞 Slow incident response due to manual reporting

### The Solution: AI-Driven Proactive Traffic Control

This system replaces the human bottleneck with a computer vision pipeline that:

| Old Way | This System |
|---|---|
| Fixed-timer signals | Density-adaptive green duration |
| No emergency detection | Instant SOS + forced GREEN on emergency |
| Manual monitoring | Automated 24/7 video analysis |
| No data logging | Per-frame CSV + session analytics graph |
| Silent failures | Telegram alerts with photo evidence |

---

## 🧠 How Does It Solve It?

### Step-by-step Pipeline

```
Video Frame
    │
    ▼
┌─────────────┐
│  YOLOv8m    │  ← Detects all objects, filters to vehicle classes
│  Inference  │    (car, truck, bus, motorcycle) at ≥30% confidence
└──────┬──────┘
       │  bounding boxes + class + confidence
       ▼
┌─────────────┐
│  DeepSort   │  ← Associates detections across frames
│  Tracker    │    Each vehicle gets a unique persistent Track ID
└──────┬──────┘
       │  confirmed tracks with IDs
       ▼
┌─────────────┐
│   Speed     │  ← Pixel displacement × metres_per_pixel × FPS × 3.6
│ Estimation  │    Smoothed over a 5-frame rolling window
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Emergency Check    │  ← Is class in EMERGENCY_VEHICLES list?
│  + Signal Control   │    YES → force signal GREEN + send Telegram alert
└──────┬──────────────┘
       │                  NO  → density-based timer (4 / 7 / 10 s)
       ▼
┌─────────────┐
│  HUD Draw   │  ← Overlays stats, signal state, bounding boxes,
│  + Display  │    speed labels on each frame using OpenCV
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Analytics  │  ← Appends row to CSV; after session ends,
│  & Logging  │    renders 4-panel matplotlib graph + saves PNG
└─────────────┘
```

### Key Technical Decisions

| Decision | Reason |
|---|---|
| **YOLOv8m** (medium) | Best balance of accuracy and speed on CPU/MPS |
| **DeepSort** tracker | Handles occlusion and re-ID better than simple IoU matching |
| **5-frame speed smoothing** | Eliminates jitter from single-frame pixel noise |
| **Telegram async thread** | Alerts never block the video loop — fire-and-forget |
| **MacOSX/Qt backend probe** | Avoids tkinter crash on Python 3.14 (no `_tkinter`) |
| **Window-close detection** | `cv2.WND_PROP_VISIBLE` ensures graph always shows on close |

---

## ✨ Features at a Glance

| Feature | Detail |
|---|---|
| 🤖 **YOLOv8 Detection** | Real-time multi-class vehicle detection |
| 📍 **DeepSort Tracking** | Persistent unique IDs across frames |
| 🌡️ **Speed Estimation** | Per-vehicle km/h with rolling-average smoothing |
| 🚦 **Adaptive Signal** | Green time scales with density (4 / 7 / 10 s) |
| 🚨 **Emergency Detection** | Trucks/buses trigger SOS + forced GREEN |
| 📱 **Telegram Alerts** | Text alert **+ photo snapshot** in seconds |
| 📸 **Auto Snapshots** | Emergency frames saved to `outputs/` as JPEG |
| 🖥️ **HUD Overlay** | Semi-transparent panel: counts, FPS, signal, speed |
| 📊 **CSV Logging** | Per-frame stats → `outputs/analytics_*.csv` |
| 📈 **Session Report** | 4-panel graph (count / confidence / FPS / emergencies) |
| 🖼️ **Image Mode** | Single-image detection + confidence bar chart |
| ⌨️ **Hotkeys** | `q` quit · `s` manual snapshot · close window → graph |

---

## 📁 Project Structure

```
emergency_vehicle/
│
├── video_detection.py      # 🎬 Main pipeline — detection, tracking, HUD, alerts, analytics
├── Image_detection.py      # 🖼️  Static image detection + confidence bar chart
├── config.py               # ⚙️  All tunable parameters (loads .env automatically)
│
├── requirements.txt        # 📦 Python dependencies
├── .env.example            # 🔑 Credential template  ← copy this to .env
├── .env                    # 🔒 Your actual secrets  (gitignored — never commit!)
├── .gitignore              # 🚫 Excludes weights, videos, venv, secrets & outputs
├── pyrightconfig.json      # 🔧 Pyright/Pyrefly config — points to .venv
│
├── .vscode/
│   └── settings.json       # 🎨 VS Code interpreter + Pyrefly settings
│
├── .venv/                  # 🐍 Virtual environment (gitignored)
├── yolov8m.pt              # 🤖 YOLOv8-medium weights (gitignored, auto-downloaded)
│
├── v1.mp4 … v6.mp4         # 🎥 Sample traffic videos (gitignored — too large)
├── t1.jpg … t8.jpg         # 🖼️  Sample images for Image_detection.py
│
└── outputs/                # 📂 Auto-created on first run
    ├── output_*.mp4        #    Annotated video (requires --save flag)
    ├── analytics_*.csv     #    Per-frame statistics
    ├── emergency_*.jpg     #    Auto-saved emergency snapshots
    ├── session_report_*.png#    4-panel analytics graph
    └── traffic_system.log  #    Full runtime log
```

> **Note:** `yolov8m.pt` and `*.mp4` files are excluded from git (too large).
> The model downloads automatically on first run via `ultralytics`.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/emergency_vehicle.git
cd emergency_vehicle
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 Weights
The model downloads automatically on first run. Or manually:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 5. Configure Telegram Credentials
```bash
cp .env.example .env
# Open .env and fill in:
#   TELEGRAM_BOT_TOKEN = your_bot_token
#   TELEGRAM_CHAT_ID   = your_chat_id
```

**How to get your credentials:**
- Message **@BotFather** on Telegram → `/newbot` → copy the token
- Message **@userinfobot** on Telegram → copy your numeric chat ID
- Send `/start` to your bot once so it can message you

### 6. Run Video Detection
```bash
# Interactive prompt — enter video path when asked
python video_detection.py

# Save annotated output video
python video_detection.py --save
```

### 7. Run Image Detection
```bash
# Edit the image_path variable in Image_detection.py, then:
python Image_detection.py
```

---

## 🎮 Usage Guide

### Video Detection (`video_detection.py`)

When launched, a prompt appears in the terminal:
```
=======================================================
  Smart Traffic System — Video Source Setup
=======================================================
  Enter a video file path to analyse.
  Examples:  v1.mp4   |   /Users/you/traffic.mp4
  Type  'webcam'  to use your system camera.
  Press  Enter  to use the DEFAULT video (config.py).
=======================================================
  Video path / webcam / [Enter for default]: _
```

| Input | Action |
|---|---|
| `v1.mp4` or full path | Opens that video file |
| *(blank Enter)* | Uses `DEFAULT_VIDEO` from `config.py` |
| `webcam` | Opens system camera (index 0) |

**Hotkeys while the video window is open:**

| Key | Action |
|---|---|
| `q` | Quit — closes window and shows session graph |
| `s` | Manually save a snapshot to `outputs/` |
| *(close window)* | Same as quitting — graph always displays |

### Image Detection (`Image_detection.py`)

1. Open the file and set `image_path` to your image
2. Run the script
3. An annotated OpenCV window opens showing bounding boxes
4. A matplotlib bar chart shows confidence scores per detection

---

## ⚙️ Configuration Reference

All parameters are in [`config.py`](config.py). Sensitive credentials come from `.env`.

### Detection & Tracking

| Parameter | Default | Description |
|---|---|---|
| `DETECTION_CONF` | `0.30` | YOLO confidence threshold — lower = more detections, more false positives |
| `TRACK_CONF` | `0.50` | Minimum confidence to pass a detection to DeepSort |
| `MIN_BOX_AREA` | `1500` | Drop bounding boxes smaller than this (px²) — removes tiny noise |
| `MAX_AGE` | `30` | Frames a lost track survives before deletion |
| `N_INIT` | `2` | Consecutive detections needed before a track is "confirmed" |

### Speed Estimation

| Parameter | Default | Description |
|---|---|---|
| `METER_PER_PIXEL` | `0.07` | **Calibrate this for your camera/scene!** Metres per pixel |
| `SPEED_SMOOTH_WINDOW` | `5` | Rolling-average window size for per-track speed |

> **Calibrating `METER_PER_PIXEL`**: Measure a known real-world distance in the scene (e.g. a lane width = 3.5 m). Count how many pixels that spans in the frame. `METER_PER_PIXEL = 3.5 / pixel_count`.

### Vehicle Classes

| Parameter | Default | Description |
|---|---|---|
| `VEHICLES` | `['car','truck','bus','motorcycle']` | Classes that enter the tracking pipeline |
| `EMERGENCY_VEHICLES` | `['truck','bus']` | Classes that trigger SOS alert + force GREEN |

### Signal & Alerts

| Parameter | Default | Description |
|---|---|---|
| `EMERGENCY_ALERT_COOLDOWN` | `5` | Minimum seconds between two Telegram alerts |
| Green duration (dynamic) | 4 / 7 / 10 s | Based on vehicle count: ≤5 → 4 s, ≤10 → 7 s, >10 → 10 s |

---

## 🖥️ HUD Layout

```
┌──────────────────────────────────────────────┐
│ SMART TRAFFIC SYSTEM   │        🟢 GREEN      │
│ ─────────────────────  │                      │
│ Vehicles  : 10         │                      │
│ Cars      : 7          │                      │
│ Trucks    : 2          │                      │
│ Buses     : 1          │                      │
│ Motorbikes: 0          │                      │
│ Total Seen: 24         │                      │
│ Frame     : #187       │                      │
│                                               │
│   [ bounding boxes + labels + speed km/h ]   │
│                                               │
│ Avg Conf: 0.82         │       FPS: 7.3       │
│ ─────────────────────────────────────────── │
│  🚨 EMERGENCY — TRUCK   |   15.1 km/h  |  Signal: GREEN (forced)  │
└──────────────────────────────────────────────┘
```

- **Top-left panel** — live per-type counts and total unique vehicles seen
- **Top-right** — traffic signal state (GREEN circle / RED circle)
- **Bottom-left** — average detection confidence
- **Bottom-right** — live FPS counter
- **Bottom banner** (red, only on emergency) — vehicle type, speed, forced signal

---

## 📊 Session Analytics Report

After every run (whether you quit normally, press `q`, or close the window), a **4-panel graph** is generated and displayed:

```
┌────────────────────┬────────────────────┐
│  Vehicle Count     │  Avg Confidence    │
│  per Frame         │  per Frame         │
├────────────────────┼────────────────────┤
│  Processing FPS    │  Emergency Events  │
│  (with avg line)   │  (binary timeline) │
└────────────────────┴────────────────────┘
```

Saved to `outputs/session_report_YYYYMMDD_HHMMSS.png`.

---

## 📱 Telegram Alert Format

```
🚨 SOS ALERT 🚨

⚠️ EMERGENCY VEHICLE DETECTED!

Vehicle Type : TRUCK
Speed        : 15.1 km/h
Total Vehicles: 11
Frame        : #142
Time         : 12:48:53
```

A JPEG snapshot of the exact frame is attached to every alert.
Alerts are throttled by `EMERGENCY_ALERT_COOLDOWN` (default 5 s) to prevent spam.

---

## 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| Graph doesn't show after video | The system auto-picks the best backend; if only `Agg` is available the PNG opens via macOS `open`. Check `outputs/` folder. |
| `Cannot find module 'cv2'` in VS Code | Select the `.venv` interpreter: `Cmd+Shift+P` → **Python: Select Interpreter** → choose `.venv` |
| Video file not found | Enter the full absolute path or make sure the file is in the project directory |
| Telegram not sending | Check token + chat ID in `.env`; ensure you sent `/start` to your bot |
| Low FPS | Switch to `yolov8n.pt` (nano model) in `config.py` for faster inference |
| Speed looks wrong | Calibrate `METER_PER_PIXEL` in `config.py` for your specific camera angle |
| `ModuleNotFoundError` on run | Activate venv: `source .venv/bin/activate` then `pip install -r requirements.txt` |
| Too many false emergency alerts | Add only true emergency classes to `EMERGENCY_VEHICLES` (remove `truck`/`bus` if needed) |

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Real-time object detection (vehicles) |
| [DeepSort Realtime](https://github.com/levan92/deep_sort_realtime) | Multi-object tracking with re-identification |
| [OpenCV](https://opencv.org/) | Video capture, frame rendering, HUD drawing |
| [python-telegram-bot](https://python-telegram-bot.org/) | Async Telegram Bot API for SOS alerts |
| [Matplotlib](https://matplotlib.org/) | Session analytics graph generation |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Secure credential loading from `.env` |
| [NumPy](https://numpy.org/) | Array math underpinning CV operations |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with ❤️ — YOLOv8 + DeepSort + OpenCV + Telegram*
