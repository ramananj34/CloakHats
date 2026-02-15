import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import json
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = Path('./raw_capture')
OUTPUT_DIR = Path('./data/drone_footage')
FRAMES_DIR = OUTPUT_DIR / 'frames'
MASKS_DIR = OUTPUT_DIR / 'masks'

ALTITUDE_FT = 25.0
ALTITUDE_M = ALTITUDE_FT * 0.3048  # 7.62 m

MIN_MASK_AREA = 500

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

logger.info("Loading YOLOv8m for person detection...")
yolo = YOLO('yolov8m.pt')

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
    if mask.sum() < MIN_MASK_AREA:
        logger.warning(f"  SKIP {img_path.name}: mask too small ({mask.sum()} px)")
        skipped['no_mask'] += 1
        continue

    results = yolo.predict(frame, conf=0.25, classes=[0], verbose=False)
    if len(results[0].boxes) == 0:
        logger.warning(f"  SKIP {img_path.name}: no person detected")
        skipped['no_person'] += 1
        continue
    best = results[0].boxes.conf.argmax()
    bbox = results[0].boxes.xyxy[best].cpu().numpy().tolist()

    bx1, by1, bx2, by2 = [int(b) for b in bbox]
    mask_in_bbox = mask[by1:by2, bx1:bx2].sum()
    if mask_in_bbox < MIN_MASK_AREA:
        logger.warning(f"  SKIP {img_path.name}: hat mask outside person bbox")
        skipped['no_mask'] += 1
        continue

    frame_id = f"frame_{frame_idx:04d}"
    frame_file = f"{frame_id}.png"
    mask_file = f"{frame_id}_mask.png"
    cv2.imwrite(str(FRAMES_DIR / frame_file), frame)
    cv2.imwrite(str(MASKS_DIR / mask_file), mask)

    camera_pitch = abs(vert_angle)
    heading = horiz_angle if horiz_angle is not None else 0.0

    annotations['frames'].append({
        'frame_id': frame_id,
        'image_path': f"frames/{frame_file}",
        'mask_path': f"masks/{mask_file}",
        'person_bbox': [round(b, 1) for b in bbox],
        'drone': {
            'camera_pitch': round(camera_pitch, 1),
            'heading': round(heading, 1),
            'altitude_meters': round(ALTITUDE_M, 2)
        }
    })

    logger.info(f"  OK {img_path.name} -> {frame_id}  pitch={camera_pitch:.1f}  heading={heading:.1f}  bbox=[{bx1},{by1},{bx2},{by2}]")
    frame_idx += 1

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
    logger.info(f"Pitch  range: [{min(pitches):.1f}, {max(pitches):.1f}] deg")
    logger.info(f"Heading range: [{min(headings):.1f}, {max(headings):.1f}] deg")
    if max(pitches) - min(pitches) < 5.0:
        logger.warning("Low pitch diversity — all frames at similar angle")
    if max(headings) - min(headings) < 30.0:
        logger.warning("Low heading diversity — consider more mannequin rotations")