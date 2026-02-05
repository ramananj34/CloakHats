import json
from pathlib import Path

def preprocess_dataset(video_dir, output_dir, fps=1):
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    frames_dir = output_dir / 'frames'
    masks_dir = output_dir / 'masks'
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    frame_idx = 0
    
    #Process each video
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.mov'))
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                #Segment green hat
                mask = segment_green_hat(frame)
                
                # Skip if no green detected
                if mask.sum() < 100:
                    count += 1
                    continue
                    
                #Save frame and mask
                frame_name = f'frame_{frame_idx:06d}.png'
                cv2.imwrite(str(frames_dir / frame_name), frame)
                cv2.imwrite(str(masks_dir / f'frame_{frame_idx:06d}_mask.png'), mask)
                
                #Get hat bounding box from mask
                ys, xs = np.where(mask > 0)
                hat_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                
                #TODO: Run person detector to get person bounding box
                #For now, estimate from hat position
                hat_cy = (hat_bbox[1] + hat_bbox[3]) // 2
                hat_cx = (hat_bbox[0] + hat_bbox[2]) // 2
                hat_h = hat_bbox[3] - hat_bbox[1]
                
                #Person is roughly 6x taller than hat
                person_h = hat_h * 6
                person_w = person_h // 3
                
                person_bbox = [
                    hat_cx - person_w // 2,
                    hat_cy - hat_h,
                    hat_cx + person_w // 2,
                    hat_cy + person_h - hat_h
                ]
                
                annotations.append({
                    'frame': frame_name,
                    'hat_bbox': hat_bbox,
                    'person_bbox': person_bbox,
                    'source_video': video_path.name,
                })
                
                frame_idx += 1
                
            count += 1
            
        cap.release()
        
    #Save annotations
    with open(output_dir / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
        
    logger.info(f"Processed {frame_idx} frames from {len(video_files)} videos")
    logger.info(f"Saved to {output_dir}")

preprocess_dataset('/raw_videos', '/data/drone_footage', fps=1)