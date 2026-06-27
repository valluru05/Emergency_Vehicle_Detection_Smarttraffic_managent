"""
config.py — Smart Traffic System Configuration
=============================================
All tunable parameters live here.
Sensitive credentials (Telegram token/chat ID) are loaded from a .env file.

Quick setup:
    cp .env.example .env
    # then fill in your credentials in .env
"""

import os
from dotenv import load_dotenv

# Load .env if present (ignored silently if absent)
load_dotenv()

# ==============================
# TELEGRAM
# ==============================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID_HERE")

# ==============================
# MODEL & VIDEO
# ==============================
MODEL_PATH    = os.getenv("MODEL_PATH",    "yolov8m.pt")
DEFAULT_VIDEO = os.getenv("DEFAULT_VIDEO", "v1.mp4")

# ==============================
# DETECTION THRESHOLDS
# ==============================
DETECTION_CONF = 0.30   # YOLO confidence threshold passed to model()
TRACK_CONF     = 0.50   # Minimum confidence to pass detection to tracker
MIN_BOX_AREA   = 1500   # Ignore bounding boxes smaller than this (px²)

# ==============================
# VEHICLE CLASSES
# ==============================
VEHICLES           = ["car", "truck", "bus", "motorcycle"]
EMERGENCY_VEHICLES = ["truck", "bus"]   # Treated as emergency → force GREEN signal

# ==============================
# DEEP SORT TRACKER
# ==============================
MAX_AGE = 30   # Frames to retain a lost track before deleting
N_INIT  = 2    # Consecutive detections required before a track is confirmed

# ==============================
# SPEED ESTIMATION
# ==============================
METER_PER_PIXEL     = 0.07  # Scene calibration: metres per pixel (adjust per camera)
SPEED_SMOOTH_WINDOW = 5     # Rolling window size for per-track speed smoothing

# ==============================
# SIGNAL & ALERT TIMING
# ==============================
EMERGENCY_ALERT_COOLDOWN = 5   # Seconds between consecutive Telegram SOS alerts

# ==============================
# OUTPUT
# ==============================
OUTPUT_DIR = "outputs"   # All generated files (video, CSV, snapshots, report) go here
