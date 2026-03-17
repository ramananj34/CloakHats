### THIS WAS MADE WITH AI

"""
preprocess_drone_footage.py

Preprocesses DJI Mini 5 Pro drone footage + decoded PhantomHelp flight log CSV
into a dataset for the CloakHat adversarial patch pipeline.

INPUTS (all in RAW_DIR):
    - 1+ MP4 video files (DJI_YYYYMMDDHHMMSS_XXXX_D.MP4)
    - 1 decoded flight log CSV from PhantomHelp

OUTPUTS (in OUTPUT_DIR):
    data/drone_footage/
        frames/          frame_XXXX.png
        hat_masks/        frame_XXXX_hat.png      (green hat binary mask)
        sweater_masks/    frame_XXXX_sweater.png   (pink sweater binary mask)
        annotations.json

USAGE:
    python preprocess_drone_footage.py

    Or override paths:
    python preprocess_drone_footage.py --raw ./RAW --output ./data/drone_footage

Requires: opencv-python, numpy, pandas, ultralytics, sahi, Pillow
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import argparse
import math
import sys
import re

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — edit these if needed
# ═══════════════════════════════════════════════════════════════

# Paths (overridable via CLI args)
RAW_DIR = Path('./RAW')
OUTPUT_DIR = Path('./data/drone_footage')

# Frame extraction
EXTRACT_FPS = 5                   # frames per second to extract
MIN_FLIGHT_HEIGHT_FT = 3.0        # ignore frames where drone is on ground

# Mannequin
MANNEQUIN_HEIGHT_FT = 6.0
MANNEQUIN_HEIGHT_M = MANNEQUIN_HEIGHT_FT * 0.3048

# Person detection
PERSON_CONF_THRESHOLD = 0.45      # YOLO confidence for person detection

# Mask segmentation — GREEN HAT (HSV ranges, OpenCV uses H:0-180, S:0-255, V:0-255)
GREEN_LOWER_1 = np.array([30, 40, 40])
GREEN_UPPER_1 = np.array([90, 255, 255])

# Mask segmentation — PINK/MAGENTA SWEATER
# Hot pink spans both ends of the hue wheel
PINK_LOWER_1 = np.array([145, 100, 50])    # magenta side
PINK_UPPER_1 = np.array([180, 255, 255])
PINK_LOWER_2 = np.array([0, 100, 50])      # red-pink wrap
PINK_UPPER_2 = np.array([12, 255, 255])

# Mask quality thresholds
MIN_HAT_MASK_AREA = 200           # minimum pixels for hat mask
MIN_SWEATER_MASK_AREA = 300       # minimum pixels for sweater mask
MIN_HAT_COVERAGE = 0.005          # hat area / person bbox area

# SAHI settings for person detection
SAHI_SLICE_SIZE = 960
SAHI_OVERLAP = 0.3

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# GEODESIC UTILITIES
# ═══════════════════════════════════════════════════════════════

def haversine_meters(lat1, lon1, lat2, lon2):
    """Compute distance in meters between two GPS points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_degrees(from_lat, from_lon, to_lat, to_lon):
    """Compute bearing in degrees (0-360, 0=North, 90=East)
    FROM one point TO another.
    
    For the pipeline azimuth we want: bearing from mannequin TO drone,
    i.e. 'the drone is to the [N/E/S/W] of the mannequin'.
    """
    phi1 = math.radians(from_lat)
    phi2 = math.radians(to_lat)
    dlam = math.radians(to_lon - from_lon)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    brg = math.degrees(math.atan2(x, y))
    return brg % 360


def get_mannequin_gps_from_first_frame(df, segment):
    """Extract mannequin GPS from the drone position at the START of the first video.
    
    User confirmed: at frame 0 of video 0003 the drone is directly above the mannequin.
    So drone lat/lon at that flyTime ≈ mannequin lat/lon.
    """
    fly_time = segment['start_flyTime']
    time_diffs = (df['flyTime_s'] - fly_time).abs()
    nearest_idx = time_diffs.idxmin()
    row = df.iloc[nearest_idx]
    
    lat = float(row['OSD.latitude'])
    lon = float(row['OSD.longitude'])
    height_ft = float(row.get('OSD.height [ft]', 0))
    gimbal_pitch = float(row.get('GIMBAL.pitch', 0))
    
    if lat == 0.0 or lon == 0.0 or math.isnan(lat) or math.isnan(lon):
        logger.error("Mannequin GPS from first frame is invalid (0,0 or NaN)!")
        logger.error("Check that the first video segment aligns correctly with the flight log.")
        sys.exit(1)
    
    logger.info(f"  MANNEQUIN GPS (from drone @ first frame of first video):")
    logger.info(f"    Lat: {lat:.10f}")
    logger.info(f"    Lon: {lon:.10f}")
    logger.info(f"    At flyTime: {fly_time:.1f}s (log row {nearest_idx})")
    logger.info(f"    Drone altitude at that moment: {height_ft:.1f} ft ({height_ft*0.3048:.1f} m)")
    logger.info(f"    Gimbal pitch at that moment:   {gimbal_pitch:.1f}°")
    
    if height_ft < 3:
        logger.warning("    ⚠ Drone appears to be on the ground — GPS may be takeoff point, not mannequin!")
    if abs(gimbal_pitch) < 30:
        logger.warning("    ⚠ Gimbal is near-horizontal — drone may not be directly above mannequin!")
    
    return lat, lon


# ═══════════════════════════════════════════════════════════════
# FLIGHT LOG PARSING
# ═══════════════════════════════════════════════════════════════

def load_flight_log(csv_path):
    """Load and parse the PhantomHelp decoded flight log CSV."""
    logger.info(f"Loading flight log: {csv_path}")
    
    # PhantomHelp CSVs start with 'sep=,' line
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    
    skiprows = 1 if first_line.startswith('sep=') else 0
    df = pd.read_csv(csv_path, skiprows=skiprows, low_memory=False)
    
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {len(df.columns)}")
    
    # Verify critical columns exist
    required = [
        'OSD.flyTime [s]', 'OSD.latitude', 'OSD.longitude',
        'OSD.height [ft]', 'GIMBAL.pitch', 'GIMBAL.yaw [360]',
        'OSD.yaw [360]', 'CAMERA.isVideo'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"  MISSING COLUMNS: {missing}")
        logger.error(f"  Available columns with 'GIMBAL': {[c for c in df.columns if 'GIMBAL' in c.upper()]}")
        logger.error(f"  Available columns with 'CAMERA': {[c for c in df.columns if 'CAMERA' in c.upper()]}")
        sys.exit(1)
    
    # Parse flyTime as float
    df['flyTime_s'] = pd.to_numeric(df['OSD.flyTime [s]'], errors='coerce')
    
    # Parse absolute timestamps for fallback alignment
    try:
        df['abs_datetime'] = pd.to_datetime(
            df['CUSTOM.date [local]'].astype(str) + ' ' + df['CUSTOM.updateTime [local]'].astype(str),
            format='mixed', dayfirst=False
        )
        logger.info(f"  Time range: {df['abs_datetime'].min()} to {df['abs_datetime'].max()}")
    except Exception as e:
        logger.warning(f"  Could not parse absolute timestamps: {e}")
        df['abs_datetime'] = pd.NaT
    
    # Log flight summary
    max_height = pd.to_numeric(df['OSD.height [ft]'], errors='coerce').max()
    max_fly_time = df['flyTime_s'].max()
    logger.info(f"  Max height: {max_height:.1f} ft")
    logger.info(f"  Total fly time: {max_fly_time:.1f} s ({max_fly_time/60:.1f} min)")
    
    # Log gimbal pitch range
    gimbal_pitch = pd.to_numeric(df['GIMBAL.pitch'], errors='coerce')
    logger.info(f"  Gimbal pitch range: [{gimbal_pitch.min():.1f}, {gimbal_pitch.max():.1f}] deg")
    
    # Find home point
    home_lat = pd.to_numeric(df['HOME.latitude'], errors='coerce').dropna()
    home_lon = pd.to_numeric(df['HOME.longitude'], errors='coerce').dropna()
    if len(home_lat) > 0:
        logger.info(f"  Home point: ({home_lat.iloc[0]:.7f}, {home_lon.iloc[0]:.7f})")
    
    return df


def find_video_segments(df):
    """Find when each video recording starts/stops in the flight log.
    
    Returns list of dicts: {start_flyTime, end_flyTime, filename}
    """
    logger.info("Searching for video recording segments in flight log...")
    
    is_video = df['CAMERA.isVideo'].astype(str).str.strip().str.lower()
    is_recording = (is_video == 'true') | (is_video == '1')
    
    # Find transitions: False→True = start, True→False = stop
    segments = []
    recording = False
    start_idx = None
    
    for i in range(len(df)):
        if is_recording.iloc[i] and not recording:
            # Start of recording
            recording = True
            start_idx = i
        elif not is_recording.iloc[i] and recording:
            # End of recording
            recording = False
            filename = ''
            # Look for filename in the segment
            seg_filenames = df.iloc[start_idx:i]['CAMERA.filename'].dropna().unique()
            seg_filenames = [str(f).strip() for f in seg_filenames if str(f).strip() and str(f).strip().lower() != 'nan']
            if seg_filenames:
                filename = seg_filenames[0]
            
            segments.append({
                'start_flyTime': df['flyTime_s'].iloc[start_idx],
                'end_flyTime': df['flyTime_s'].iloc[i-1],
                'start_idx': start_idx,
                'end_idx': i - 1,
                'filename': filename,
                'duration_s': df['flyTime_s'].iloc[i-1] - df['flyTime_s'].iloc[start_idx],
            })
    
    # Handle case where recording is still on at end of log
    if recording and start_idx is not None:
        seg_filenames = df.iloc[start_idx:]['CAMERA.filename'].dropna().unique()
        seg_filenames = [str(f).strip() for f in seg_filenames if str(f).strip() and str(f).strip().lower() != 'nan']
        filename = seg_filenames[0] if seg_filenames else ''
        segments.append({
            'start_flyTime': df['flyTime_s'].iloc[start_idx],
            'end_flyTime': df['flyTime_s'].iloc[-1],
            'start_idx': start_idx,
            'end_idx': len(df) - 1,
            'filename': filename,
            'duration_s': df['flyTime_s'].iloc[-1] - df['flyTime_s'].iloc[start_idx],
        })
    
    logger.info(f"  Found {len(segments)} recording segment(s) in flight log")
    for i, seg in enumerate(segments):
        logger.info(f"    Segment {i}: flyTime {seg['start_flyTime']:.1f}–{seg['end_flyTime']:.1f}s "
                     f"({seg['duration_s']:.1f}s) filename='{seg['filename']}'")
    
    return segments


def match_videos_to_segments(video_files, segments, df):
    """Match each MP4 file to a flight log segment.
    
    Strategy:
    1. Try matching by CAMERA.filename
    2. Fall back to matching by duration + order
    
    Returns: list of (video_path, segment_dict) tuples
    """
    logger.info(f"Matching {len(video_files)} video files to {len(segments)} flight log segments...")
    
    matches = []
    unmatched_videos = list(video_files)
    unmatched_segments = list(range(len(segments)))
    
    # Strategy 1: filename matching
    for vid_path in video_files:
        for seg_idx in unmatched_segments:
            seg = segments[seg_idx]
            if seg['filename'] and seg['filename'] in vid_path.stem:
                matches.append((vid_path, seg))
                unmatched_videos.remove(vid_path)
                unmatched_segments.remove(seg_idx)
                logger.info(f"  MATCHED (filename): {vid_path.name} ↔ segment {seg_idx}")
                break
    
    # Strategy 2: match remaining by order (both sorted by time)
    if unmatched_videos and unmatched_segments:
        logger.info(f"  {len(unmatched_videos)} unmatched videos, trying duration/order matching...")
        
        # Get video durations
        remaining_with_duration = []
        for vid_path in sorted(unmatched_videos):
            cap = cv2.VideoCapture(str(vid_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            remaining_with_duration.append((vid_path, duration))
            logger.info(f"    Video {vid_path.name}: {duration:.1f}s ({frame_count} frames @ {fps:.0f}fps)")
        
        remaining_segs = [(i, segments[i]) for i in sorted(unmatched_segments)]
        for seg_i, seg in remaining_segs:
            logger.info(f"    Segment {seg_i}: {seg['duration_s']:.1f}s")
        
        # Match by order (assumes videos and segments are time-ordered)
        for (vid_path, vid_dur), (seg_idx, seg) in zip(remaining_with_duration, remaining_segs):
            dur_diff = abs(vid_dur - seg['duration_s'])
            logger.info(f"  ORDER MATCH: {vid_path.name} ({vid_dur:.1f}s) ↔ segment {seg_idx} ({seg['duration_s']:.1f}s) [Δ={dur_diff:.1f}s]")
            if dur_diff > max(30, vid_dur * 0.3):
                logger.warning(f"    ⚠ Duration mismatch > 30s or 30% — alignment may be off!")
            matches.append((vid_path, seg))
    
    logger.info(f"  Total matches: {len(matches)}")
    return matches


def lookup_telemetry(df, fly_time_s, mannequin_lat, mannequin_lon):
    """Look up the nearest flight log row for a given flyTime.
    
    Uses mannequin GPS to compute true lateral distance and azimuth,
    instead of inferring from gimbal angle (which assumes camera is aimed at target).
    
    Returns dict with all telemetry values, or None if no valid data.
    """
    # Find nearest row by flyTime
    time_diffs = (df['flyTime_s'] - fly_time_s).abs()
    nearest_idx = time_diffs.idxmin()
    row = df.iloc[nearest_idx]
    
    # Check temporal proximity (should be within 0.15s at 10Hz)
    actual_diff = time_diffs.iloc[nearest_idx]
    if actual_diff > 1.0:
        return None  # too far from any log entry
    
    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return v if not math.isnan(v) else default
        except (ValueError, TypeError):
            return default
    
    gimbal_pitch = safe_float(row.get('GIMBAL.pitch'), 0.0)
    gimbal_yaw_360 = safe_float(row.get('GIMBAL.yaw [360]'), 0.0)
    height_ft = safe_float(row.get('OSD.height [ft]'), 0.0)
    drone_lat = safe_float(row.get('OSD.latitude'), 0.0)
    drone_lon = safe_float(row.get('OSD.longitude'), 0.0)
    drone_yaw_360 = safe_float(row.get('OSD.yaw [360]'), 0.0)
    home_dist_ft = safe_float(row.get('HOME.distance [ft]'), 0.0)
    home_lat = safe_float(row.get('HOME.latitude'), 0.0)
    home_lon = safe_float(row.get('HOME.longitude'), 0.0)
    
    height_m = height_ft * 0.3048
    home_dist_m = home_dist_ft * 0.3048
    
    # camera_pitch: the pipeline uses 90 = directly above, 0 = horizon
    # GIMBAL.pitch: -90 = straight down, 0 = horizon
    camera_pitch = abs(gimbal_pitch)
    
    # ── TRUE lateral distance via GPS haversine ──
    # This is accurate regardless of where the camera is pointed
    if drone_lat != 0.0 and drone_lon != 0.0:
        lateral_distance_m = haversine_meters(drone_lat, drone_lon, mannequin_lat, mannequin_lon)
    else:
        lateral_distance_m = 0.0
    
    # ── TRUE 3D distance = sqrt(lateral² + height²) ──
    distance_3d_m = math.sqrt(lateral_distance_m**2 + height_m**2)
    
    # ── GEOMETRIC ELEVATION = angle from mannequin up to drone ──
    # This is what PyTorch3D's look_at_view_transform(elev=...) needs.
    # It's the angle of the camera's POSITION, not where the camera is aimed.
    # 90° = drone directly above, 0° = drone at same height as mannequin.
    # Different from gimbal_pitch when camera isn't aimed at mannequin.
    if distance_3d_m > 0.1:
        elevation_to_mannequin = math.degrees(math.atan2(height_m, lateral_distance_m))
    else:
        elevation_to_mannequin = 90.0  # directly above
    
    # ── AZIMUTH = bearing from mannequin TO drone ──
    # This tells the renderer "the drone is viewing from this compass direction"
    # Independent of camera aim — just based on positions
    if drone_lat != 0.0 and drone_lon != 0.0 and lateral_distance_m > 0.5:
        azimuth = bearing_degrees(mannequin_lat, mannequin_lon, drone_lat, drone_lon)
    else:
        # Drone is basically on top of mannequin — azimuth doesn't matter much
        azimuth = gimbal_yaw_360  # fallback to gimbal yaw
    
    return {
        'gimbal_pitch_raw': gimbal_pitch,
        'camera_pitch': round(elevation_to_mannequin, 2),  # ← GEOMETRIC, for the renderer
        'gimbal_pitch_abs': round(camera_pitch, 2),         # ← raw gimbal angle, for reference
        'heading': round(azimuth, 2),
        'gimbal_yaw_360': round(gimbal_yaw_360, 2),
        'drone_yaw_360': round(drone_yaw_360, 2),
        'altitude_ft': round(height_ft, 2),
        'altitude_meters': round(height_m, 3),
        'lateral_distance_meters': round(lateral_distance_m, 3),
        'distance_3d_meters': round(distance_3d_m, 3),
        'home_distance_ft': round(home_dist_ft, 2),
        'home_distance_m': round(home_dist_m, 3),
        'drone_lat': drone_lat,
        'drone_lon': drone_lon,
        'home_lat': home_lat,
        'home_lon': home_lon,
        'mannequin_lat': mannequin_lat,
        'mannequin_lon': mannequin_lon,
        'fly_time_s': round(fly_time_s, 3),
        'log_time_offset_s': round(actual_diff, 4),
    }

# ═══════════════════════════════════════════════════════════════
# SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def segment_green_hat(frame_bgr):
    """Segment the green hat from a BGR frame. Returns binary mask (0/255)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER_1, GREEN_UPPER_1)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    # Gentle dilation to recover edges
    dilate_kernel = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    
    # Re-isolate largest component after dilation (dilation can bridge noise)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels2 > 2:
        largest2 = 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels2 == largest2, 255, 0).astype(np.uint8)
    
    return mask


def segment_pink_sweater(frame_bgr):
    """Segment the pink/magenta sweater from a BGR frame. Returns binary mask (0/255)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # Pink spans both ends of the hue wheel
    mask1 = cv2.inRange(hsv, PINK_LOWER_1, PINK_UPPER_1)
    mask2 = cv2.inRange(hsv, PINK_LOWER_2, PINK_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    # Gentle dilation
    dilate_kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    
    # Re-isolate largest after dilation
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels2 > 2:
        largest2 = 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels2 == largest2, 255, 0).astype(np.uint8)
    
    return mask


def constrain_mask_to_bbox(mask, bbox, expand=1.5):
    """Zero out mask pixels far outside an expanded person bounding box."""
    x1, y1, x2, y2 = [int(b) for b in bbox]
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * (expand - 1) / 2)
    pad_y = int(bh * (expand - 1) / 2)
    h, w = mask.shape[:2]
    
    bbox_region = np.zeros_like(mask)
    ry1 = max(y1 - pad_y, 0)
    ry2 = min(y2 + pad_y, h)
    rx1 = max(x1 - pad_x, 0)
    rx2 = min(x2 + pad_x, w)
    bbox_region[ry1:ry2, rx1:rx2] = 255
    
    return cv2.bitwise_and(mask, bbox_region)


def mask_overlaps_bbox(mask, bbox, min_overlap_ratio=0.0):
    """Check if any mask pixels fall inside the person bounding box."""
    x1, y1, x2, y2 = [int(b) for b in bbox]
    h, w = mask.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    roi = mask[y1:y2, x1:x2]
    return np.count_nonzero(roi) > 0

# ═══════════════════════════════════════════════════════════════
# PERSON DETECTION
# ═══════════════════════════════════════════════════════════════

# Path to VisDrone fine-tuned model (optional — runs without it)
VISDRONE_MODEL_PATH = 'runs/detect/runs/detect/yolov8m-visdrone/weights/best.pt'


def init_detectors():
    """Initialize detectors for maximum person recall.
    
    Two complementary detectors:
      1. COCO YOLO direct   — fast, good close range, standard perspectives
      2. VisDrone SAHI tiled — aerial-optimized + tiled = best far-range aerial
    
    If VisDrone model doesn't exist yet (still training), falls back to COCO SAHI.
    Any person found by either source counts. NMS merges duplicates.
    """
    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    
    logger.info("Initializing person detectors...")
    
    detectors = {}
    
    # ── COCO YOLOv8m direct (always available) ──
    detectors['coco_yolo'] = YOLO('yolov8m.pt')
    logger.info("  ✓ COCO YOLOv8m direct loaded")
    
    # ── VisDrone fine-tuned (optional but preferred) ──
    visdrone_path = Path(VISDRONE_MODEL_PATH)
    has_visdrone = visdrone_path.exists()
    
    if has_visdrone:
        # VisDrone SAHI — the best combo for aerial small-person detection
        detectors['visdrone_sahi'] = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(visdrone_path),
            confidence_threshold=PERSON_CONF_THRESHOLD,
            device='cuda:0'
        )
        logger.info(f"  ✓ VisDrone SAHI loaded from {visdrone_path}")
    else:
        # Fallback: COCO SAHI (worse for aerial, but better than nothing)
        detectors['coco_sahi'] = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='yolov8m.pt',
            confidence_threshold=PERSON_CONF_THRESHOLD,
            device='cuda:0'
        )
        logger.info(f"  ✓ COCO SAHI loaded (fallback — VisDrone not found at {visdrone_path})")
        logger.info(f"    Train VisDrone for better aerial detection: python train_visdrone.py")
    
    logger.info(f"  Active detectors: {list(detectors.keys())}")
    return detectors


def compute_iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0.0


def nms_merge(detections, iou_threshold=0.5):
    """Merge overlapping detections from multiple sources.
    
    Two boxes with IoU > threshold are the same person — keep the
    higher-confidence one. This prevents multiple detectors finding
    the same mannequin from counting as 2+ persons.
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    dets = sorted(detections, key=lambda d: d['conf'], reverse=True)
    keep = []
    
    while dets:
        best = dets.pop(0)
        keep.append(best)
        # Remove anything that overlaps enough with the kept detection
        remaining = []
        for d in dets:
            if compute_iou(best['bbox'], d['bbox']) < iou_threshold:
                remaining.append(d)  # different person, keep for next round
        dets = remaining
    
    return keep


def detect_persons(frame_bgr, detectors, frame_path=None):
    """Detect persons using two complementary detectors, then merge via NMS.
    
    Pass 1: COCO YOLO direct   — fast, good close range (classes=[0] person)
    Pass 2: VisDrone SAHI tiled — aerial-optimized, good far range (classes 0+1)
            Falls back to COCO SAHI if VisDrone model not available.
    
    Returns list of dicts: {bbox: [x1,y1,x2,y2], conf: float, source: str}
    """
    from sahi.predict import get_sliced_prediction
    
    all_detections = []
    
    # Ensure frame is on disk for SAHI
    if frame_path is None:
        frame_path = '/tmp/_sahi_temp.png'
        cv2.imwrite(frame_path, frame_bgr)
    
    # ── Pass 1: COCO YOLO direct (always runs) ──
    coco_model = detectors['coco_yolo']
    results = coco_model.predict(frame_bgr, conf=PERSON_CONF_THRESHOLD, classes=[0], verbose=False)
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            all_detections.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'conf': float(box.conf.cpu()),
                'source': 'coco_yolo',
            })
    
    # ── Pass 2: SAHI tiled (VisDrone or COCO fallback) ──
    if 'visdrone_sahi' in detectors:
        sahi_model = detectors['visdrone_sahi']
        # VisDrone person classes: 0=pedestrian, 1=people
        person_class_ids = {0, 1}
        sahi_source = 'visdrone_sahi'
    else:
        sahi_model = detectors['coco_sahi']
        # COCO person class: 0
        person_class_ids = {0}
        sahi_source = 'coco_sahi'
    
    sahi_result = get_sliced_prediction(
        str(frame_path), sahi_model,
        slice_height=SAHI_SLICE_SIZE,
        slice_width=SAHI_SLICE_SIZE,
        overlap_height_ratio=SAHI_OVERLAP,
        overlap_width_ratio=SAHI_OVERLAP,
        verbose=0,
    )
    for pred in sahi_result.object_prediction_list:
        if pred.category.id in person_class_ids:
            bbox = pred.bbox.to_xyxy()
            all_detections.append({
                'bbox': [float(b) for b in bbox],
                'conf': pred.score.value,
                'source': sahi_source,
            })
    
    # ── Merge: NMS deduplication ──
    merged = nms_merge(all_detections, iou_threshold=0.5)
    
    return merged

# ═══════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_all(raw_dir, output_dir):
    """Main entry point: process all videos + flight log into dataset."""
    
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    frames_dir = output_dir / 'frames'
    hat_masks_dir = output_dir / 'hat_masks'
    sweater_masks_dir = output_dir / 'sweater_masks'
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    hat_masks_dir.mkdir(parents=True, exist_ok=True)
    sweater_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Find input files ──
    video_files = sorted(raw_dir.glob('*.MP4')) + sorted(raw_dir.glob('*.mp4'))
    csv_files = sorted(raw_dir.glob('FlightRecord*.csv'))
    
    if not video_files:
        logger.error(f"No MP4 files found in {raw_dir}")
        sys.exit(1)
    if not csv_files:
        logger.error(f"No FlightRecord CSV found in {raw_dir}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("PREPROCESSING DRONE FOOTAGE")
    logger.info("=" * 70)
    logger.info(f"  Raw directory:    {raw_dir.resolve()}")
    logger.info(f"  Output directory: {output_dir.resolve()}")
    logger.info(f"  Video files:      {len(video_files)}")
    for v in video_files:
        logger.info(f"    {v.name} ({v.stat().st_size / 1e6:.0f} MB)")
    logger.info(f"  Flight log:       {csv_files[0].name}")
    logger.info(f"  Extract FPS:      {EXTRACT_FPS}")
    logger.info(f"  Mannequin height: {MANNEQUIN_HEIGHT_FT} ft ({MANNEQUIN_HEIGHT_M:.2f} m)")
    
    # ── Load flight log ──
    df = load_flight_log(csv_files[0])
    
    # ── Find video segments in flight log ──
    segments = find_video_segments(df)
    
    # ── Filter out tiny accidental segments (brief record/stop with no real MP4) ──
    MIN_SEGMENT_DURATION = 10.0  # seconds
    original_count = len(segments)
    segments = [s for s in segments if s['duration_s'] >= MIN_SEGMENT_DURATION]
    if len(segments) < original_count:
        removed = original_count - len(segments)
        logger.info(f"  Filtered out {removed} segment(s) shorter than {MIN_SEGMENT_DURATION}s (accidental record/stop)")
        logger.info(f"  Remaining segments: {len(segments)}")
        for i, seg in enumerate(segments):
            logger.info(f"    Segment {i}: flyTime {seg['start_flyTime']:.1f}–{seg['end_flyTime']:.1f}s ({seg['duration_s']:.1f}s)")
    
    # ── Match videos to segments ──
    if len(segments) == 0:
        logger.warning("No recording segments found via CAMERA.isVideo column!")
        logger.warning("Falling back to timestamp-based alignment from MP4 filenames...")
        # Fallback: parse timestamps from MP4 filenames and flight log
        segments = create_fallback_segments(video_files, df)
    
    matches = match_videos_to_segments(video_files, segments, df)
    
    if not matches:
        logger.error("Could not match any video to flight log segments!")
        logger.error("Check that the flight log CSV matches the video files.")
        sys.exit(1)
    
    # ── Extract mannequin GPS from first frame of first video ──
    # User confirmed: drone is directly above mannequin at frame 0 of the first video (0003)
    # Sort matches by filename so 0003 comes first
    matches.sort(key=lambda m: m[0].name)
    first_vid, first_seg = matches[0]
    logger.info("")
    logger.info(f"Extracting mannequin GPS from first frame of {first_vid.name}...")
    mannequin_lat, mannequin_lon = get_mannequin_gps_from_first_frame(df, first_seg)
    
    # Sanity check: compute distance from home to mannequin
    home_lat_vals = pd.to_numeric(df['HOME.latitude'], errors='coerce').dropna()
    home_lon_vals = pd.to_numeric(df['HOME.longitude'], errors='coerce').dropna()
    if len(home_lat_vals) > 0:
        home_to_mannequin = haversine_meters(
            home_lat_vals.iloc[0], home_lon_vals.iloc[0],
            mannequin_lat, mannequin_lon
        )
        logger.info(f"  Distance home→mannequin: {home_to_mannequin:.1f} m ({home_to_mannequin*3.281:.1f} ft)")
    
    # ── Initialize detectors (both YOLO and SAHI) ──
    detectors = init_detectors()
    
    # ── Process each video ──
    all_frames = []
    global_frame_idx = 0
    
    skip_stats = {
        'low_altitude': 0,
        'no_person': 0,
        'no_mask_at_all': 0,
        'no_telemetry': 0,
        'total_extracted': 0,
        'total_kept': 0,
        'kept_hat_only': 0,
        'kept_sweater_only': 0,
        'kept_both': 0,
    }
    
    for vid_path, segment in matches:
        logger.info("")
        logger.info("─" * 60)
        logger.info(f"PROCESSING: {vid_path.name}")
        logger.info(f"  Segment flyTime: {segment['start_flyTime']:.1f}–{segment['end_flyTime']:.1f}s")
        logger.info("─" * 60)
        
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            logger.error(f"  Cannot open video: {vid_path}")
            continue
        
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_duration = total_frames_in_vid / vid_fps if vid_fps > 0 else 0
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"  Resolution: {vid_width}x{vid_height}")
        logger.info(f"  FPS: {vid_fps:.2f}")
        logger.info(f"  Total frames: {total_frames_in_vid}")
        logger.info(f"  Duration: {vid_duration:.1f}s")
        
        # Calculate frame interval for desired extraction rate
        frame_interval = int(round(vid_fps / EXTRACT_FPS))
        frames_to_extract = total_frames_in_vid // frame_interval
        logger.info(f"  Extracting every {frame_interval}th frame → ~{frames_to_extract} frames at {EXTRACT_FPS}fps")
        
        vid_frame_idx = 0
        extracted_from_vid = 0
        kept_from_vid = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame
            if vid_frame_idx % frame_interval != 0:
                vid_frame_idx += 1
                continue
            
            skip_stats['total_extracted'] += 1
            
            # Compute flyTime for this frame
            frame_time_in_video = vid_frame_idx / vid_fps
            frame_fly_time = segment['start_flyTime'] + frame_time_in_video
            
            # Progress logging
            extracted_from_vid += 1
            if extracted_from_vid % 50 == 0:
                logger.info(f"  ... processed {extracted_from_vid}/{frames_to_extract} frames, kept {kept_from_vid}")
            
            # ── Look up telemetry ──
            telemetry = lookup_telemetry(df, frame_fly_time, mannequin_lat, mannequin_lon)
            if telemetry is None:
                skip_stats['no_telemetry'] += 1
                vid_frame_idx += 1
                continue
            
            # ── Skip if drone is on the ground ──
            if telemetry['altitude_ft'] < MIN_FLIGHT_HEIGHT_FT:
                skip_stats['low_altitude'] += 1
                vid_frame_idx += 1
                continue
            
            # ── Save frame temporarily for SAHI ──
            frame_id = f"frame_{global_frame_idx:05d}"
            frame_path = frames_dir / f"{frame_id}.png"
            cv2.imwrite(str(frame_path), frame)
            
            # ── Person detection (YOLO + SAHI union) ──
            persons = detect_persons(frame, detectors, frame_path=str(frame_path))
            
            if len(persons) == 0:
                skip_stats['no_person'] += 1
                frame_path.unlink(missing_ok=True)
                vid_frame_idx += 1
                continue

            # ── Find the person wearing the green hat or pink sweater ──
            quick_hat = segment_green_hat(frame)
            quick_sweater = segment_pink_sweater(frame)
            
            best_person = None
            best_overlap = 0
            for p in persons:
                bx1, by1, bx2, by2 = [int(b) for b in p['bbox']]
                h, w = quick_hat.shape[:2]
                bx1, by1 = max(bx1, 0), max(by1, 0)
                bx2, by2 = min(bx2, w), min(by2, h)
                hat_roi = quick_hat[by1:by2, bx1:bx2]
                sweater_roi = quick_sweater[by1:by2, bx1:bx2]
                overlap = np.count_nonzero(hat_roi) + np.count_nonzero(sweater_roi)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person = p
            
            if best_person is None or best_overlap < MIN_HAT_MASK_AREA:
                skip_stats['no_person'] += 1
                frame_path.unlink(missing_ok=True)
                vid_frame_idx += 1
                continue
            
            person = best_person
            person_bbox = person['bbox']
            person_conf = person['conf']
            
            # ── Compute person bbox area (used for coverage) ──
            bx1, by1, bx2, by2 = [int(b) for b in person_bbox]
            bbox_area = max((bx2 - bx1) * (by2 - by1), 1)
            
            # ── Segment green hat ──
            hat_mask = quick_hat
            hat_mask = constrain_mask_to_bbox(hat_mask, person_bbox, expand=1.5)
            hat_area = np.count_nonzero(hat_mask)
            hat_on_person = mask_overlaps_bbox(hat_mask, person_bbox) if hat_area > 0 else False
            hat_coverage = hat_area / bbox_area
            hat_usable = (hat_area >= MIN_HAT_MASK_AREA
                          and hat_on_person
                          and hat_coverage >= MIN_HAT_COVERAGE)
            
            # ── Segment pink sweater ──
            sweater_mask = quick_sweater
            sweater_mask = constrain_mask_to_bbox(sweater_mask, person_bbox, expand=1.3)
            sweater_area = np.count_nonzero(sweater_mask)
            sweater_on_person = mask_overlaps_bbox(sweater_mask, person_bbox) if sweater_area > 0 else False
            sweater_coverage = sweater_area / bbox_area
            sweater_usable = (sweater_area >= MIN_SWEATER_MASK_AREA
                              and sweater_on_person
                              and sweater_coverage >= MIN_HAT_COVERAGE)  # same threshold
            
            # ── GATE: need at least ONE usable mask ──
            if not hat_usable and not sweater_usable:
                skip_stats['no_mask_at_all'] += 1
                frame_path.unlink(missing_ok=True)
                vid_frame_idx += 1
                continue
            
            # ── Track what we're keeping ──
            if hat_usable and sweater_usable:
                skip_stats['kept_both'] += 1
            elif hat_usable:
                skip_stats['kept_hat_only'] += 1
            else:
                skip_stats['kept_sweater_only'] += 1
            
            # ── FRAME ACCEPTED — save everything ──
            # Frame already saved, now save masks (even empty ones — pipeline checks usable flags)
            hat_mask_path = hat_masks_dir / f"{frame_id}_hat.png"
            sweater_mask_path = sweater_masks_dir / f"{frame_id}_sweater.png"
            
            cv2.imwrite(str(hat_mask_path), hat_mask)
            cv2.imwrite(str(sweater_mask_path), sweater_mask)
            
            # Build annotation
            annotation = {
                'frame_id': frame_id,
                'image_path': f"frames/{frame_id}.png",
                'mask_path': f"hat_masks/{frame_id}_hat.png",  # ← backward compat with old DroneDataset
                'hat_mask_path': f"hat_masks/{frame_id}_hat.png",
                'sweater_mask_path': f"sweater_masks/{frame_id}_sweater.png",
                'person_bbox': [round(b, 1) for b in person_bbox],
                'person_conf': round(person_conf, 3),
                'hat_usable': hat_usable,
                'sweater_usable': sweater_usable,
                'drone': {
                    **telemetry,
                    'hat_coverage': round(hat_coverage, 4),
                    'sweater_coverage': round(sweater_coverage, 4),
                    'video_file': vid_path.name,
                    'frame_in_video': vid_frame_idx,
                },
            }
            
            all_frames.append(annotation)
            global_frame_idx += 1
            kept_from_vid += 1
            skip_stats['total_kept'] += 1
            
            vid_frame_idx += 1
        
        cap.release()
        logger.info(f"  ✓ {vid_path.name}: extracted {extracted_from_vid}, kept {kept_from_vid}")
    
    # ═══════════════════════════════════════════════════════════
    # POST-PROCESSING & ANNOTATIONS
    # ═══════════════════════════════════════════════════════════
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("POST-PROCESSING")
    logger.info("=" * 70)
    
    # Compute distance-to-scale calibration from the data
    if all_frames:
        distances = [f['drone']['distance_3d_meters'] for f in all_frames]
        altitudes = [f['drone']['altitude_meters'] for f in all_frames]
        laterals = [f['drone']['lateral_distance_meters'] for f in all_frames]
        min_dist = min(distances)
        max_dist = max(distances)
        logger.info(f"  3D distance range:     {min_dist:.1f}–{max_dist:.1f} m")
        logger.info(f"  Lateral distance range: {min(laterals):.1f}–{max(laterals):.1f} m")
        logger.info(f"  Altitude range:         {min(altitudes):.1f}–{max(altitudes):.1f} m")
        logger.info(f"  Mannequin GPS (known):  ({mannequin_lat:.10f}, {mannequin_lon:.10f})")
    else:
        min_dist, max_dist = 5.0, 50.0
    
    # Also keep the old altitude_to_scale for backwards compatibility
    altitudes_m = [f['drone']['altitude_meters'] for f in all_frames] if all_frames else [5, 50]
    
    # Build final annotations
    annotations = {
        'frames': all_frames,
        'metadata': {
            'mannequin_height_ft': MANNEQUIN_HEIGHT_FT,
            'mannequin_height_m': MANNEQUIN_HEIGHT_M,
            'distance_to_scale': {
                'min_distance': round(max(min_dist - 1, 1.0), 2),
                'max_distance': round(max_dist + 5, 2),
                'min_scale': 0.3,
                'max_scale': 1.2,
            },
            'altitude_to_scale': {
                'min_altitude': round(max(min(altitudes_m) - 1, 1.0), 2),
                'max_altitude': round(max(altitudes_m) + 5, 2),
                'min_scale': 0.3,
                'max_scale': 1.2,
            },
            'mannequin_lat': mannequin_lat,
            'mannequin_lon': mannequin_lon,
            'mannequin_gps_source': f'drone position at frame 0 of {first_vid.name}',
            'home_lat': all_frames[0]['drone']['home_lat'] if all_frames else 0.0,
            'home_lon': all_frames[0]['drone']['home_lon'] if all_frames else 0.0,
            'extract_fps': EXTRACT_FPS,
            'green_hsv_range': [GREEN_LOWER_1.tolist(), GREEN_UPPER_1.tolist()],
            'pink_hsv_ranges': [
                [PINK_LOWER_1.tolist(), PINK_UPPER_1.tolist()],
                [PINK_LOWER_2.tolist(), PINK_UPPER_2.tolist()],
            ],
        }
    }
    
    ann_path = output_dir / 'annotations.json'
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ═══════════════════════════════════════════════════════════
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total frames extracted:      {skip_stats['total_extracted']}")
    logger.info(f"  Total frames KEPT:           {skip_stats['total_kept']}")
    logger.info(f"    ├─ hat + sweater usable:   {skip_stats['kept_both']}")
    logger.info(f"    ├─ hat only usable:        {skip_stats['kept_hat_only']}")
    logger.info(f"    └─ sweater only usable:    {skip_stats['kept_sweater_only']}")
    logger.info(f"  Skip — low altitude:         {skip_stats['low_altitude']}")
    logger.info(f"  Skip — no person detected:   {skip_stats['no_person']}")
    logger.info(f"  Skip — no usable mask:       {skip_stats['no_mask_at_all']}")
    logger.info(f"  Skip — no telemetry match:   {skip_stats['no_telemetry']}")
    
    if all_frames:
        pitches = [f['drone']['camera_pitch'] for f in all_frames]
        headings = [f['drone']['heading'] for f in all_frames]
        hat_covs = [f['drone']['hat_coverage'] for f in all_frames]
        sweater_covs = [f['drone']['sweater_coverage'] for f in all_frames]
        laterals = [f['drone']['lateral_distance_meters'] for f in all_frames]
        
        hat_usable_frames = [f for f in all_frames if f['hat_usable']]
        sweater_usable_frames = [f for f in all_frames if f['sweater_usable']]
        
        logger.info("")
        logger.info("  TELEMETRY RANGES:")
        logger.info(f"    Geometric elev:  [{min(pitches):.1f}°, {max(pitches):.1f}°]  (90°=overhead) ← used by renderer")
        gimbal_pitches = [f['drone']['gimbal_pitch_abs'] for f in all_frames]
        logger.info(f"    Gimbal pitch:    [{min(gimbal_pitches):.1f}°, {max(gimbal_pitches):.1f}°]  (raw camera tilt, for reference)")
        logger.info(f"    Heading (azim):  [{min(headings):.1f}°, {max(headings):.1f}°]  (GPS bearing)")
        logger.info(f"    Altitude:        [{min(altitudes_m):.1f}m, {max(altitudes_m):.1f}m]")
        logger.info(f"    Lateral dist:    [{min(laterals):.1f}m, {max(laterals):.1f}m]")
        logger.info(f"    3D distance:     [{min(distances):.1f}m, {max(distances):.1f}m]")
        logger.info("")
        logger.info("  MASK USABILITY:")
        logger.info(f"    Hat usable:      {len(hat_usable_frames)}/{len(all_frames)} frames ({len(hat_usable_frames)/len(all_frames):.0%})")
        logger.info(f"    Sweater usable:  {len(sweater_usable_frames)}/{len(all_frames)} frames ({len(sweater_usable_frames)/len(all_frames):.0%})")
        
        if hat_usable_frames:
            hat_pitches = [f['drone']['camera_pitch'] for f in hat_usable_frames]
            logger.info(f"    Hat pitch range:     [{min(hat_pitches):.1f}°, {max(hat_pitches):.1f}°]")
        if sweater_usable_frames:
            sw_pitches = [f['drone']['camera_pitch'] for f in sweater_usable_frames]
            logger.info(f"    Sweater pitch range: [{min(sw_pitches):.1f}°, {max(sw_pitches):.1f}°]")
        
        logger.info("")
        logger.info("  MASK COVERAGE (all kept frames):")
        logger.info(f"    Hat coverage:     [{min(hat_covs):.1%}, {max(hat_covs):.1%}]  mean={np.mean(hat_covs):.1%}")
        logger.info(f"    Sweater coverage: [{min(sweater_covs):.1%}, {max(sweater_covs):.1%}]  mean={np.mean(sweater_covs):.1%}")
        
        # Quality warnings
        logger.info("")
        if max(pitches) - min(pitches) < 10:
            logger.warning("  ⚠ LOW PITCH DIVERSITY — all frames at similar camera angle")
        else:
            logger.info("  ✓ Good pitch diversity")
        
        if max(headings) - min(headings) < 60:
            logger.warning("  ⚠ LOW HEADING DIVERSITY — consider more orbit coverage")
        else:
            logger.info("  ✓ Good heading diversity")
        
        if len(hat_usable_frames) < 20:
            logger.warning(f"  ⚠ LOW HAT FRAME COUNT ({len(hat_usable_frames)}) — may need more overhead passes")
        else:
            logger.info(f"  ✓ Good hat frame count ({len(hat_usable_frames)})")
        
        if len(sweater_usable_frames) < 20:
            logger.warning(f"  ⚠ LOW SWEATER FRAME COUNT ({len(sweater_usable_frames)}) — may need more low-angle passes")
        else:
            logger.info(f"  ✓ Good sweater frame count ({len(sweater_usable_frames)})")
    
    logger.info("")
    logger.info(f"  Output: {output_dir.resolve()}")
    logger.info(f"  Annotations: {ann_path}")
    logger.info("=" * 70)
    
    return annotations


def create_fallback_segments(video_files, df):
    """When CAMERA.isVideo is unreliable, create segments from MP4 filenames + flight log times."""
    logger.info("Creating fallback segments from MP4 filename timestamps...")
    
    # Parse flight log start time
    if df['abs_datetime'].notna().any():
        log_start = df.loc[df['abs_datetime'].notna(), 'abs_datetime'].iloc[0]
        log_start_fly = df.loc[df['abs_datetime'].notna(), 'flyTime_s'].iloc[0]
    else:
        logger.error("Cannot determine flight log start time for fallback alignment!")
        return []
    
    segments = []
    for vid_path in sorted(video_files):
        # Parse timestamp from DJI filename: DJI_YYYYMMDDHHMMSS_XXXX_D.MP4
        match = re.search(r'DJI_(\d{14})', vid_path.name)
        if not match:
            logger.warning(f"  Cannot parse timestamp from {vid_path.name}")
            continue
        
        ts_str = match.group(1)
        vid_time = datetime.strptime(ts_str, '%Y%m%d%H%M%S')
        
        # Get video duration
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = nframes / fps if fps > 0 else 0
        cap.release()
        
        # Try multiple timezone offsets to find the best alignment
        # The flight log "[local]" label is unreliable
        # Try: same timezone, +5h (UTC vs EST), -5h (EST vs UTC)
        best_offset_hours = 0
        best_diff = float('inf')
        
        for offset_hours in [0, 5, -5, 4, -4, 6, -6]:
            adjusted_vid_time = vid_time + timedelta(hours=offset_hours)
            diff = abs((adjusted_vid_time - log_start).total_seconds())
            # Video must start AFTER flight log start, within the flight duration
            time_into_flight = (adjusted_vid_time - log_start).total_seconds() + log_start_fly
            if 0 <= time_into_flight <= df['flyTime_s'].max() and diff < best_diff:
                best_diff = diff
                best_offset_hours = offset_hours
        
        adjusted_vid_time = vid_time + timedelta(hours=best_offset_hours)
        fly_time_offset = (adjusted_vid_time - log_start).total_seconds() + log_start_fly
        
        logger.info(f"  {vid_path.name}: filename_time={vid_time}, offset={best_offset_hours:+d}h, "
                     f"flyTime={fly_time_offset:.1f}–{fly_time_offset + duration:.1f}s")
        
        segments.append({
            'start_flyTime': fly_time_offset,
            'end_flyTime': fly_time_offset + duration,
            'start_idx': 0,
            'end_idx': 0,
            'filename': vid_path.stem,
            'duration_s': duration,
        })
    
    return segments


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess DJI drone footage for CloakHat pipeline')
    parser.add_argument('--raw', type=str, default=str(RAW_DIR), help='Directory with MP4s and flight log CSV')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory for dataset')
    parser.add_argument('--fps', type=int, default=EXTRACT_FPS, help='Frame extraction rate (frames/sec)')
    parser.add_argument('--mannequin-height', type=float, default=MANNEQUIN_HEIGHT_FT, help='Mannequin height in feet')
    args = parser.parse_args()
    
    RAW_DIR = Path(args.raw)
    OUTPUT_DIR = Path(args.output)
    EXTRACT_FPS = args.fps
    MANNEQUIN_HEIGHT_FT = args.mannequin_height
    MANNEQUIN_HEIGHT_M = MANNEQUIN_HEIGHT_FT * 0.3048
    
    process_all(RAW_DIR, OUTPUT_DIR)