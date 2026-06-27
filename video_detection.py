#!/usr/bin/env python3
"""
Smart Traffic System — Enhanced v2.0
=====================================
Real-time vehicle detection, multi-object tracking, speed estimation,
adaptive signal control, emergency alerts, and session analytics.

Usage:
    python video_detection.py

    At the prompt, enter:
      - A video file path  (e.g. v1.mp4  or  /full/path/to/video.mp4)
      - Leave blank to use the DEFAULT_VIDEO set in config.py
      - Type  webcam  to use the system webcam

Hotkeys (in the display window):
    q  — Quit
    s  — Save a manual snapshot to outputs/
    (Closing the window also stops detection and shows the session graph)
"""

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import matplotlib
# Probe backends in order of preference; matplotlib.use() does NOT raise
# when a backend is unavailable — it only fails at figure-creation time.
# We therefore test each backend by trying to import its module directly.
def _select_backend() -> str:
    """Return the best available matplotlib backend."""
    candidates = [
        ("MacOSX",  "matplotlib.backends.backend_macosx"),   # native macOS, no Tk
        ("Qt5Agg",  "matplotlib.backends.backend_qt5agg"),   # Qt5
        ("GTK3Agg", "matplotlib.backends.backend_gtk3agg"),  # GTK3
        ("Agg",     None),                                   # always works (file only)
    ]
    import importlib
    for name, module in candidates:
        if module is None:
            return name   # Agg always available
        try:
            importlib.import_module(module)
            return name
        except Exception:
            continue
    return "Agg"

_backend = _select_backend()
matplotlib.use(_backend)
import matplotlib.pyplot as plt
import math
from telegram import Bot
from telegram.error import TelegramError
import threading
import asyncio
import logging
import csv
import os
import sys
from collections import deque, defaultdict
from datetime import datetime

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    MODEL_PATH, DEFAULT_VIDEO,
    DETECTION_CONF, TRACK_CONF, MIN_BOX_AREA,
    VEHICLES, EMERGENCY_VEHICLES,
    MAX_AGE, N_INIT,
    METER_PER_PIXEL, SPEED_SMOOTH_WINDOW,
    EMERGENCY_ALERT_COOLDOWN, OUTPUT_DIR,
)

# ==============================
# OUTPUT DIRECTORY
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOGGING
# ==============================
log_file = os.path.join(OUTPUT_DIR, "traffic_system.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ==============================
# USER INPUT (OpenCV / terminal)
# ==============================
print("\n" + "=" * 55)
print("  Smart Traffic System — Video Source Setup")
print("=" * 55)
print("  Enter a video file path to analyse.")
print("  Examples:  v1.mp4   |   /Users/you/traffic.mp4")
print("  Type  'webcam'  to use your system camera.")
print("  Press  Enter  to use the DEFAULT video (config.py).")
print("  (Annotated video auto-saves if you add  '--save'  flag)")
print("=" * 55)
user_input = input("  Video path / webcam / [Enter for default]: ").strip()
print()

# ==============================
# TELEGRAM HELPERS
# ==============================
async def _send_telegram(message: str, image_path: str | None = None) -> None:
    """Async: send text alert and optionally a photo to Telegram."""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as photo:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=photo,
                    caption="📸 Emergency vehicle snapshot",
                )
        log.info("Telegram alert sent ✅")
    except TelegramError as e:
        log.error(f"Telegram alert failed ❌: {e}")


def send_alert_async(message: str, image_path: str | None = None) -> None:
    """Run Telegram alert in a background thread — never blocks video loop."""
    try:
        asyncio.run(_send_telegram(message, image_path))
    except Exception as e:
        log.error(f"Alert thread error: {e}")

# ==============================
# HUD DRAWING HELPERS
# ==============================
def draw_transparent_rect(
    frame,
    x1: int, y1: int,
    x2: int, y2: int,
    color: tuple = (0, 0, 0),
    alpha: float = 0.55,
) -> None:
    """Blend a filled rectangle onto the frame for a glass-panel HUD effect."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def put_text(frame, text: str, pos: tuple, scale: float = 0.5,
             color: tuple = (255, 255, 255), thickness: int = 1) -> None:
    """Convenience wrapper for cv2.putText."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

# ==============================
# LOAD MODEL
# ==============================
log.info(f"Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
log.info("Model loaded ✅")

# ==============================
# VIDEO SOURCE
# ==============================
if user_input.lower() == "webcam":
    log.info("Input: webcam")
    cap = cv2.VideoCapture(0)
else:
    video_path = user_input if user_input else DEFAULT_VIDEO
    log.info(f"Input: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Could not open '{video_path}' — falling back to default: {DEFAULT_VIDEO}")
        cap = cv2.VideoCapture(DEFAULT_VIDEO)
        if not cap.isOpened():
            log.error("Default video also failed. Trying webcam as last resort.")
            cap = cv2.VideoCapture(0)

frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
log.info(f"Resolution: {frame_w}x{frame_h}  Source FPS: {src_fps:.1f}")

# ==============================
# OUTPUT VIDEO WRITER (optional)
# ==============================
video_writer = None
if "--save" in sys.argv:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"output_{ts}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_path, fourcc, src_fps, (frame_w, frame_h))
    log.info(f"Saving annotated video to: {out_path}")

# Name of the OpenCV display window (used to detect when user closes it)
WINDOW_NAME = "Smart Traffic System v2.0"

# ==============================
# TRACKER
# ==============================
tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT)

# ==============================
# TRAFFIC SIGNAL STATE
# ==============================
signal_state           = "RED"
last_switch_time       = time.time()
green_duration         = 5
last_emergency_alert_t = 0.0

# ==============================
# PER-TRACK STATE
# ==============================
track_positions  : dict[int, tuple[int, int]]   = {}
speed_buffers    : dict[int, deque]             = defaultdict(
    lambda: deque(maxlen=SPEED_SMOOTH_WINDOW)
)
unique_vehicle_ids: set[int] = set()

# ==============================
# SESSION ANALYTICS BUFFERS
# ==============================
vehicle_history   : list[int]   = []
confidence_history: list[float] = []
fps_history       : list[float] = []
emergency_history : list[int]   = []   # 1 = emergency frame, 0 = normal

# ==============================
# CSV ANALYTICS LOG
# ==============================
csv_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_DIR, f"analytics_{csv_ts}.csv")
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp", "frame", "vehicle_count",
    "cars", "trucks", "buses", "motorcycles",
    "avg_conf", "live_fps",
    "signal_state", "emergency_detected", "emergency_speed_kmh",
    "total_unique_vehicles",
])
log.info(f"CSV analytics: {csv_path}")

# ==============================
# RUNTIME COUNTERS
# ==============================
frame_count = 0
fps_window  = deque(maxlen=30)   # timestamps for rolling FPS calculation

log.info("Detection loop started.  Hotkeys: [q] quit  [s] manual snapshot")

# ══════════════════════════════════════════════════════════════════
# MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════════
# Create the window once so we can monitor its state
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    # ── Detect if user closed the window (works on most platforms) ──
    try:
        win_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        if win_visible < 1:
            log.info("Video window closed by user — stopping detection.")
            break
    except cv2.error:
        log.info("Video window destroyed — stopping detection.")
        break

    ok, frame = cap.read()
    if not ok:
        log.info("End of video stream — exiting loop.")
        break

    frame_count += 1
    fps_window.append(time.time())

    # --- Rolling FPS ---
    live_fps = (
        (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
        if len(fps_window) > 1 else 0.0
    )

    # ── Per-frame reset ──────────────────────────────────────────
    conf_sum          = 0.0
    det_count         = 0
    vehicle_count     = 0
    type_counts: dict[str, int] = defaultdict(int)
    emergency_detected   = False
    emergency_speed      = 0.0
    emergency_vehicle_type = ""

    # ── YOLO Inference ───────────────────────────────────────────
    results    = model(frame, conf=DETECTION_CONF)[0]
    detections = []

    for box in results.boxes:
        cls       = int(box.cls[0])
        conf      = float(box.conf[0])
        det_class = model.names[cls].lower()

        if det_class not in VEHICLES or conf < TRACK_CONF:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, det_class))
        conf_sum  += conf
        det_count += 1

    # ── DeepSort Tracking ────────────────────────────────────────
    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed() or t.det_conf is None:
            continue

        x1, y1, x2, y2 = map(int, t.to_ltrb())
        det_label = (t.get_det_class() or "vehicle").lower()

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # --- Speed (smoothed) ---
        raw_speed = 0.0
        if t.track_id in track_positions:
            px, py    = track_positions[t.track_id]
            px_dist   = math.hypot(cx - px, cy - py)
            raw_speed = (px_dist * METER_PER_PIXEL * src_fps) * 3.6

        track_positions[t.track_id] = (cx, cy)
        speed_buffers[t.track_id].append(raw_speed)
        smooth_speed = sum(speed_buffers[t.track_id]) / len(speed_buffers[t.track_id])

        # --- Unique vehicle registry ---
        unique_vehicle_ids.add(t.track_id)

        # --- Emergency classification ---
        if det_label in EMERGENCY_VEHICLES:
            display_label        = f"Emergency ({det_label})"
            box_color            = (30, 30, 230)
            emergency_detected   = True
            emergency_speed      = smooth_speed
            emergency_vehicle_type = det_label
        else:
            display_label = det_label
            box_color     = (30, 210, 80)

        type_counts[det_label] += 1
        vehicle_count          += 1

        # --- Draw bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label background pill
        label_text = f"{display_label}  {t.det_conf:.2f}  {smooth_speed:.1f} km/h"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        lbl_y = max(y1 - 4, th + 6)
        cv2.rectangle(frame,
                      (x1, lbl_y - th - 4), (x1 + tw + 6, lbl_y + 2),
                      box_color, -1)
        put_text(frame, label_text, (x1 + 3, lbl_y - 2),
                 scale=0.42, color=(255, 255, 255))

    # ── Adaptive Signal Logic ────────────────────────────────────
    current_time = time.time()

    if emergency_detected:
        signal_state     = "GREEN"
        last_switch_time = current_time

        # Throttle alerts by cooldown
        if current_time - last_emergency_alert_t > EMERGENCY_ALERT_COOLDOWN:
            # Save emergency snapshot
            snap_ts   = datetime.now().strftime("%H%M%S_%f")[:9]
            snap_path = os.path.join(OUTPUT_DIR, f"emergency_{snap_ts}.jpg")
            cv2.imwrite(snap_path, frame)
            log.info(f"Emergency snapshot saved: {snap_path}")

            alert_msg = (
                f"🚨 SOS ALERT 🚨\n\n"
                f"⚠️ EMERGENCY VEHICLE DETECTED!\n\n"
                f"Vehicle Type : {emergency_vehicle_type.upper()}\n"
                f"Speed        : {emergency_speed:.1f} km/h\n"
                f"Total Vehicles: {vehicle_count}\n"
                f"Frame        : #{frame_count}\n"
                f"Time         : {time.strftime('%H:%M:%S')}"
            )
            threading.Thread(
                target=send_alert_async,
                args=(alert_msg, snap_path),
                daemon=True,
            ).start()
            last_emergency_alert_t = current_time
    else:
        # Density-based green timing
        if vehicle_count > 10:
            green_duration = 10
        elif vehicle_count > 5:
            green_duration = 7
        else:
            green_duration = 4

        if current_time - last_switch_time > green_duration:
            signal_state     = "GREEN" if signal_state == "RED" else "RED"
            last_switch_time = current_time

    # ── HUD Overlay ──────────────────────────────────────────────
    avg_conf = conf_sum / det_count if det_count > 0 else 0.0

    # Top-left stats panel
    draw_transparent_rect(frame, 0, 0, 238, 168)
    put_text(frame, "SMART TRAFFIC SYSTEM",  (8, 17),  0.44, (0, 210, 255))
    cv2.line(frame, (8, 21), (230, 21), (0, 210, 255), 1)
    put_text(frame, f"Vehicles   : {vehicle_count}",                 (8, 40),  0.50, (255, 255, 255))
    put_text(frame, f"Cars       : {type_counts.get('car', 0)}",     (8, 60),  0.48, (140, 255, 140))
    put_text(frame, f"Trucks     : {type_counts.get('truck', 0)}",   (8, 80),  0.48, (140, 180, 255))
    put_text(frame, f"Buses      : {type_counts.get('bus', 0)}",     (8, 100), 0.48, (255, 200, 140))
    put_text(frame, f"Motorbikes : {type_counts.get('motorcycle', 0)}", (8, 120), 0.48, (255, 255, 140))
    put_text(frame, f"Total Seen : {len(unique_vehicle_ids)}",       (8, 145), 0.50, (0, 230, 255))
    put_text(frame, f"Frame      : #{frame_count}",                  (8, 163), 0.42, (160, 160, 160))

    # Bottom-left: avg confidence
    draw_transparent_rect(frame, 0, frame_h - 30, 190, frame_h, alpha=0.50)
    put_text(frame, f"Avg Conf: {avg_conf:.3f}", (8, frame_h - 10), 0.50, (255, 230, 0))

    # Top-right: traffic signal
    draw_transparent_rect(frame, frame_w - 185, 0, frame_w, 68, alpha=0.60)
    sig_color = (30, 230, 30) if signal_state == "GREEN" else (30, 30, 220)
    sig_label_color = (30, 230, 30) if signal_state == "GREEN" else (80, 80, 255)
    cv2.circle(frame, (frame_w - 28, 34), 20, sig_color, -1)
    cv2.circle(frame, (frame_w - 28, 34), 20, (255, 255, 255), 2)
    put_text(frame, signal_state, (frame_w - 168, 40), 0.72, sig_label_color, 2)

    # Bottom-right: live FPS
    draw_transparent_rect(frame, frame_w - 125, frame_h - 30, frame_w, frame_h, alpha=0.50)
    put_text(frame, f"FPS: {live_fps:.1f}", (frame_w - 115, frame_h - 10), 0.50, (0, 255, 200))

    # Emergency warning banner (bottom, full width)
    if emergency_detected:
        draw_transparent_rect(frame, 0, frame_h - 68, frame_w, frame_h - 32,
                               color=(0, 0, 160), alpha=0.65)
        banner = (f"🚨  EMERGENCY — {emergency_vehicle_type.upper()}"
                  f"   |   {emergency_speed:.1f} km/h"
                  f"   |   Signal: GREEN (forced)")
        put_text(frame, banner, (10, frame_h - 44), 0.60, (255, 90, 90), 2)

    # ── Analytics buffers ────────────────────────────────────────
    vehicle_history.append(vehicle_count)
    confidence_history.append(avg_conf)
    fps_history.append(live_fps)
    emergency_history.append(1 if emergency_detected else 0)

    # CSV row
    csv_writer.writerow([
        datetime.now().isoformat(), frame_count, vehicle_count,
        type_counts.get("car", 0), type_counts.get("truck", 0),
        type_counts.get("bus", 0), type_counts.get("motorcycle", 0),
        round(avg_conf, 4), round(live_fps, 2),
        signal_state, int(emergency_detected), round(emergency_speed, 2),
        len(unique_vehicle_ids),
    ])

    # ── Display ──────────────────────────────────────────────────
    if video_writer:
        video_writer.write(frame)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        log.info("Quit by user.")
        break
    elif key == ord("s"):
        snap_path = os.path.join(
            OUTPUT_DIR,
            f"manual_snap_{datetime.now().strftime('%H%M%S')}.jpg"
        )
        cv2.imwrite(snap_path, frame)
        log.info(f"Manual snapshot saved: {snap_path}")

# ══════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
csv_file.close()
log.info("Resources released.")

# ── Session summary ───────────────────────────────────────────────
total_emergency_frames = sum(emergency_history)
log.info(
    f"\n{'='*50}\n"
    f"  SESSION COMPLETE\n"
    f"  Frames processed : {frame_count}\n"
    f"  Unique vehicles  : {len(unique_vehicle_ids)}\n"
    f"  Emergency frames : {total_emergency_frames}\n"
    f"  Avg confidence   : {sum(confidence_history)/len(confidence_history):.3f}\n"
    f"  Avg FPS          : {sum(fps_history)/max(len(fps_history),1):.1f}\n"
    f"{'='*50}"
)

# ══════════════════════════════════════════════════════════════════
# SESSION REPORT — 4-panel analytics graph
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    f"Smart Traffic System — Session Report\n"
    f"Frames: {frame_count}  |  Unique Vehicles: {len(unique_vehicle_ids)}"
    f"  |  Emergency Frames: {total_emergency_frames}",
    fontsize=13, fontweight="bold", color="#1a1a2e",
)
fig.patch.set_facecolor("#f0f4f8")

for ax in axes.flat:
    ax.set_facecolor("#e8edf5")
    ax.grid(True, linestyle="--", alpha=0.45, color="#a0aabb")
    ax.spines[["top", "right"]].set_visible(False)

frames = range(len(vehicle_history))

# Graph 1 — Vehicle count
axes[0, 0].plot(frames, vehicle_history, color="#2196F3", linewidth=1.3, label="Count")
axes[0, 0].fill_between(frames, vehicle_history, alpha=0.15, color="#2196F3")
axes[0, 0].set_title("Vehicle Count per Frame", fontweight="bold")
axes[0, 0].set_xlabel("Frame"); axes[0, 0].set_ylabel("Vehicles")

# Graph 2 — Detection confidence
axes[0, 1].plot(frames, confidence_history, color="#4CAF50", linewidth=1.3)
axes[0, 1].fill_between(frames, confidence_history, alpha=0.15, color="#4CAF50")
axes[0, 1].set_title("Avg Detection Confidence", fontweight="bold")
axes[0, 1].set_xlabel("Frame"); axes[0, 1].set_ylabel("Confidence")
axes[0, 1].set_ylim(0, 1.05)

# Graph 3 — Processing FPS
axes[1, 0].plot(frames, fps_history, color="#FF9800", linewidth=1.3)
axes[1, 0].fill_between(frames, fps_history, alpha=0.15, color="#FF9800")
avg_fps_val = sum(fps_history) / max(len(fps_history), 1)
axes[1, 0].axhline(avg_fps_val, color="#e65100", linestyle="--",
                    linewidth=1, label=f"Avg {avg_fps_val:.1f} FPS")
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title("Processing FPS", fontweight="bold")
axes[1, 0].set_xlabel("Frame"); axes[1, 0].set_ylabel("FPS")

# Graph 4 — Emergency events
axes[1, 1].fill_between(frames, emergency_history,
                         alpha=0.45, color="#f44336", step="mid")
axes[1, 1].step(frames, emergency_history, color="#c62828",
                linewidth=1.2, where="mid")
axes[1, 1].set_title(
    f"Emergency Detections  ({total_emergency_frames} frames / {frame_count})",
    fontweight="bold"
)
axes[1, 1].set_xlabel("Frame"); axes[1, 1].set_ylabel("Emergency")
axes[1, 1].set_ylim(-0.05, 1.15)
axes[1, 1].set_yticks([0, 1]); axes[1, 1].set_yticklabels(["No", "Yes"])

plt.tight_layout(rect=[0, 0, 1, 0.94])

report_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(OUTPUT_DIR, f"session_report_{report_ts}.png")
plt.savefig(report_path, dpi=150, bbox_inches="tight")
log.info(f"Session report saved: {report_path}")

# Show the graph window.
# Interactive backends (MacOSX, Qt5Agg, …) open a live window.
# Agg is file-only, so we open the saved PNG with the OS viewer instead.
if _backend == "Agg":
    log.info(f"No interactive display — opening report image: {report_path}")
    try:
        import subprocess
        subprocess.Popen(["open", report_path])   # macOS 'open' command
    except Exception as e:
        log.warning(f"Could not open report image automatically: {e}. "
                    f"PNG saved to: {report_path}")
else:
    try:
        plt.show(block=True)
    except Exception as e:
        log.warning(f"Could not display interactive graph ({e}). "
                    f"PNG saved to: {report_path}")