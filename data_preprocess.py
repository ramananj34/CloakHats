#Made with help from AI

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import json
import logging
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = Path('./raw_capture2')
OUTPUT_DIR = Path('./data/drone_footage')
FRAMES_DIR = OUTPUT_DIR / 'frames'
MASKS_DIR = OUTPUT_DIR / 'masks'

ALTITUDE_FT = 25.0
ALTITUDE_M = ALTITUDE_FT * 0.3048

MIN_MASK_AREA = 500

# SAHI settings
SLICE_SIZE = 960
SLICE_OVERLAP = 0.3
CONF_THRESHOLD = 0.25

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

def segment_green_hat(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #Keep only largest connected component BEFORE dilation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    # Now dilate only the hat blob
    dilate_kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    
    return mask

def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif is None:
            return None, None
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'ImageDescription':
                vert, horiz = None, None
                for part in value.split('/'):
                    part = part.strip()
                    if 'vert_angle_deg' in part:
                        vert = float(part.split('=')[1].strip())
                    elif 'horiz_angle_deg' in part:
                        horiz = float(part.split('=')[1].strip())
                return vert, horiz
    except Exception as e:
        logger.warning(f"  EXIF read failed for {image_path.name}: {e}")
    return None, None

def sahi_detect_person(image_path, detection_model):
    """Run SAHI sliced detection and return best person bbox and confidence."""
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=SLICE_OVERLAP,
        overlap_width_ratio=SLICE_OVERLAP,
        verbose=0,
    )
    best_conf = 0.0
    best_bbox = None
    for pred in result.object_prediction_list:
        if pred.category.id == 0 and pred.score.value > best_conf:
            best_conf = pred.score.value
            best_bbox = pred.bbox.to_xyxy()
    return best_bbox, best_conf

logger.info("Loading YOLOv8m with SAHI...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8m.pt',
    confidence_threshold=CONF_THRESHOLD,
    device='cuda:0'
)

annotations = {
    'frames': [],
    'metadata': {
        'altitude_to_scale': {
            'min_altitude': 5.0,
            'max_altitude': 50.0,
            'min_scale': 0.3,
            'max_scale': 1.2
        }
    }
}

image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
raw_images = sorted([f for f in RAW_DIR.iterdir() if f.suffix.lower() in image_exts])
logger.info(f"Found {len(raw_images)} images in {RAW_DIR}")
logger.info(f"SAHI config: slice={SLICE_SIZE}, overlap={SLICE_OVERLAP}, conf={CONF_THRESHOLD}")

skipped = {'no_meta': 0, 'no_read': 0, 'no_mask': 0, 'no_person': 0}
frame_idx = 0

for img_path in raw_images:
    vert_angle, horiz_angle = extract_metadata(img_path)
    if vert_angle is None:
        logger.warning(f"  SKIP {img_path.name}: no metadata")
        skipped['no_meta'] += 1
        continue

    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.warning(f"  SKIP {img_path.name}: unreadable")
        skipped['no_read'] += 1
        continue

    mask = segment_green_hat(frame)
    if np.count_nonzero(mask) < MIN_MASK_AREA:
        logger.warning(f"  SKIP {img_path.name}: mask too small ({np.count_nonzero(mask)} px)")
        skipped['no_mask'] += 1
        continue

    # SAHI sliced detection
    best_bbox, best_conf = sahi_detect_person(img_path, detection_model)
    if best_bbox is None:
        logger.warning(f"  SKIP {img_path.name}: no person detected (SAHI)")
        skipped['no_person'] += 1
        continue

    bbox = [float(b) for b in best_bbox]
    bx1, by1, bx2, by2 = [int(b) for b in bbox]

    # Zero out mask outside 1.5x expanded person bbox
    bw, bh = bx2 - bx1, by2 - by1
    pad_x, pad_y = int(bw * 0.25), int(bh * 0.25)
    h, w = mask.shape[:2]
    bbox_mask = np.zeros_like(mask)
    bbox_mask[max(by1-pad_y,0):min(by2+pad_y,h), max(bx1-pad_x,0):min(bx2+pad_x,w)] = 255
    mask = cv2.bitwise_and(mask, bbox_mask)
    
    mask_in_bbox = np.count_nonzero(mask[by1:by2, bx1:bx2])
    if mask_in_bbox < MIN_MASK_AREA:
        logger.warning(f"  SKIP {img_path.name}: hat mask outside person bbox")
        skipped['no_mask'] += 1
        continue

    # Coverage
    bbox_area = (bx2 - bx1) * (by2 - by1)
    coverage = mask_in_bbox / bbox_area if bbox_area > 0 else 0.0

    camera_pitch = abs(vert_angle)
    heading = horiz_angle if horiz_angle is not None else 0.0

    frame_id = f"frame_{frame_idx:04d}"
    frame_file = f"{frame_id}.png"
    mask_file = f"{frame_id}_mask.png"
    cv2.imwrite(str(FRAMES_DIR / frame_file), frame)
    cv2.imwrite(str(MASKS_DIR / mask_file), mask)

    annotations['frames'].append({
        'frame_id': frame_id,
        'image_path': f"frames/{frame_file}",
        'mask_path': f"masks/{mask_file}",
        'person_bbox': [round(b, 1) for b in bbox],
        'drone': {
            'camera_pitch': round(camera_pitch, 1),
            'heading': round(heading, 1),
            'altitude_meters': round(ALTITUDE_M, 2),
            'hat_coverage': round(coverage, 4)
        }
    })

    logger.info(f"  OK {img_path.name} -> {frame_id}  conf={best_conf:.2f}  cov={coverage:.1%}  pitch={camera_pitch:.1f}  heading={heading:.1f}")
    frame_idx += 1

# Baseline verification — SAHI at conf=0.5
logger.info("=" * 50)
logger.info("Verifying baseline detection (SAHI @ conf=0.5)...")
baseline_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8m.pt',
    confidence_threshold=0.5,
    device='cuda:0'
)

pass_full = 0
pass_crop = 0
total = len(annotations['frames'])

# Direct YOLO on crop (no SAHI needed — crop is already 640x640)
from ultralytics import YOLO as YOLO_direct
yolo_crop = YOLO_direct('yolov8m.pt')

for entry in annotations['frames']:
    frame_path = OUTPUT_DIR / entry['image_path']
    
    # Full frame SAHI check
    bbox_check, conf_check = sahi_detect_person(frame_path, baseline_model)
    if bbox_check is not None:
        pass_full += 1
    
    # Crop check (simulates training pipeline)
    frame = cv2.imread(str(frame_path))
    bbox = entry['person_bbox']
    x1, y1, x2, y2 = [int(b) for b in bbox]
    bw, bh = x2 - x1, y2 - y1
    side = int(max(bw, bh) * 1.5)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = frame.shape[:2]
    cx1 = max(cx - side // 2, 0)
    cy1 = max(cy - side // 2, 0)
    cx2 = min(cx + side // 2, w)
    cy2 = min(cy + side // 2, h)
    crop = frame[cy1:cy2, cx1:cx2]
    crop_resized = cv2.resize(crop, (640, 640))
    
    results_crop = yolo_crop.predict(crop_resized, conf=0.5, classes=[0], verbose=False)
    if len(results_crop[0].boxes) > 0:
        pass_crop += 1

logger.info(f"BASELINE @ conf=0.5 ({total} frames):")
logger.info(f"  SAHI full-frame: {pass_full}/{total} ({pass_full/total:.1%})")
logger.info(f"  Direct crop 640: {pass_crop}/{total} ({pass_crop/total:.1%})")

ann_path = OUTPUT_DIR / 'annotations.json'
with open(ann_path, 'w') as f:
    json.dump(annotations, f, indent=2)

logger.info("=" * 50)
logger.info(f"Processed: {frame_idx} frames")
logger.info(f"Skipped:   {sum(skipped.values())} total")
for reason, count in skipped.items():
    if count > 0:
        logger.info(f"  {reason}: {count}")
logger.info(f"Output:    {OUTPUT_DIR}")
logger.info(f"Annotations: {ann_path}")

if annotations['frames']:
    pitches = [f['drone']['camera_pitch'] for f in annotations['frames']]
    headings = [f['drone']['heading'] for f in annotations['frames']]
    coverages = [f['drone']['hat_coverage'] for f in annotations['frames']]
    logger.info(f"Pitch    range: [{min(pitches):.1f}, {max(pitches):.1f}] deg")
    logger.info(f"Heading  range: [{min(headings):.1f}, {max(headings):.1f}] deg")
    logger.info(f"Coverage range: [{min(coverages):.1%}, {max(coverages):.1%}]")
    logger.info(f"Coverage mean:  {np.mean(coverages):.1%}")
    if max(pitches) - min(pitches) < 5.0:
        logger.warning("Low pitch diversity — all frames at similar angle")
    if max(headings) - min(headings) < 30.0:
        logger.warning("Low heading diversity — consider more mannequin rotations")
    if np.mean(coverages) < 0.10:
        logger.warning("MEAN COVERAGE BELOW 10% — hat may be too small for effective attack")
    if np.mean(coverages) >= 0.15:
        logger.info("GOOD: Coverage in target range for bucket hat attack")