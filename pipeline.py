import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Import our custom tracker
from tracker import GMCByteTracker

def get_color(idx):
    """
    Generate a distinct BGR color using HSV space (Golden Angle distribution).
    """
    h = (idx * 137.5) % 180  # OpenCV HSV H is 0-179
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in bgr)

def draw_sleek_box(img, box, track_id, color, class_name):
    """
    Draw a premium bounding box with rounded corners and a transparent ID tag.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Draw thin bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # Bounding box corners (sleek visual highlight)
    len_corner = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
    if len_corner > 2:
        # Top-left corner
        cv2.line(img, (x1, y1), (x1 + len_corner, y1), color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + len_corner), color, 3, cv2.LINE_AA)
        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - len_corner, y1), color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + len_corner), color, 3, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1 + len_corner, y2), color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - len_corner), color, 3, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - len_corner, y2), color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - len_corner), color, 3, cv2.LINE_AA)

    # Draw label tag
    label = f"ID: {track_id}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    thickness = 1
    (w_text, h_text), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Ensure tag doesn't go off-screen
    tag_y1 = max(0, y1 - h_text - 8)
    tag_y2 = tag_y1 + h_text + 8
    tag_x1 = x1
    tag_x2 = x1 + w_text + 10
    
    # Draw background tag rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), (20, 20, 20), -1)
    # Blend overlay for transparent background
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    
    # Draw label text
    cv2.putText(img, label, (tag_x1 + 5, tag_y2 - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_fade_tail(img, history, color):
    """
    Draw a fading trajectory tail for tracked objects.
    """
    if len(history) < 2:
        return
    for i in range(len(history) - 1):
        pt1 = history[i]
        pt2 = history[i+1]
        
        # Newer points are at the end (closer to current position)
        age_ratio = i / (len(history) - 1)  # 0 to 1
        
        # Calculate thickness and color fading
        thickness = int(1 + 2 * age_ratio)
        color_intensity = tuple(int(c * age_ratio) for c in color)
        
        cv2.line(img, pt1, pt2, color_intensity, thickness, cv2.LINE_AA)

def process_sequence(seq_dir, model_name, imgsz, conf, tracker_config, output_path, limit_frames):
    # Check if the model is from Hugging Face
    if "/" in model_name:
        print(f"Downloading model '{model_name}' weights from Hugging Face...")
        import importlib
        huggingface_hub = None
        try:
            huggingface_hub = importlib.import_module("huggingface_hub")
        except ImportError:
            print("huggingface_hub package is not installed. Attempting to install it via pip...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                importlib.invalidate_caches()
                huggingface_hub = importlib.import_module("huggingface_hub")
            except Exception as inst_err:
                print(f"Failed to install huggingface_hub: {inst_err}")
        
        if huggingface_hub is not None:
            try:
                local_weights = huggingface_hub.hf_hub_download(repo_id=model_name, filename="best.pt")
                print(f"Model weights downloaded to: {local_weights}")
                model_name = local_weights
            except Exception as e:
                print(f"Warning: Failed to download from Hugging Face: {e}")
                print("Will attempt to load as local file path.")
        else:
            print("Warning: huggingface_hub is not available. Cannot download model weights.")

    print(f"Loading YOLO model: {model_name}...")
    model = YOLO(model_name)
    
    # Map class IDs to names for the VisDrone model
    # Usually: 0: pedestrian, 1: people
    class_names = {0: "pedestrian", 1: "people"}
    target_classes = [0, 1]  # Persons
    
    # Initialize our custom GMCByteTracker
    tracker = GMCByteTracker(
        track_thresh=conf,
        track_buffer=tracker_config.get("track_buffer", 30),
        match_thresh=tracker_config.get("match_thresh", 0.8)
    )
    
    # Check if input is a video file or a sequence folder
    is_video = os.path.isfile(seq_dir) and seq_dir.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    
    if is_video:
        cap = cv2.VideoCapture(seq_dir)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {seq_dir}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_in = fps_in if fps_in > 0 else 25.0
        
        if limit_frames > 0 and limit_frames < total_frames:
            num_frames = limit_frames
        else:
            num_frames = total_frames
            
        print(f"Processing video: {os.path.basename(seq_dir)}")
        print(f"Frames to process: {num_frames} | Resolution: {width}x{height} | FPS: {fps_in}")
    else:
        # Find all image frames in the sequence directory
        frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if not frame_files:
            raise ValueError(f"No image files or supported video found at: {seq_dir}")
            
        if limit_frames > 0:
            frame_files = frame_files[:limit_frames]
            print(f"Limiting frame processing to first {len(frame_files)} frames.")
            
        num_frames = len(frame_files)
        first_frame_path = os.path.join(seq_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, _ = first_frame.shape
        fps_in = 25.0
        
        print(f"Processing sequence: {os.path.basename(seq_dir)}")
        print(f"Frames: {num_frames} | Resolution: {width}x{height}")
        
    # Set up VideoWriter
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
    
    total_det_time = 0
    total_track_time = 0
    total_vis_time = 0
    processed_frames = 0
    
    # Active track statistics
    unique_track_ids = set()
    
    # Loop over all frames
    for f_idx in tqdm(range(num_frames), desc="Processing Frames"):
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame_file = frame_files[f_idx]
            frame_path = os.path.join(seq_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
        processed_frames += 1
            
        # 1. Object Detection (YOLO)
        t_det_start = time.time()
        results = model(frame, imgsz=imgsz, conf=conf, classes=target_classes, verbose=False)
        t_det_end = time.time()
        total_det_time += (t_det_end - t_det_start)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            clses = results[0].boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, clses):
                detections.append([box[0], box[1], box[2], box[3], score, int(cls)])
                
        # 2. Update Tracker
        t_track_start = time.time()
        online_tracks = tracker.update(detections, frame)
        t_track_end = time.time()
        total_track_time += (t_track_end - t_track_start)
        
        # 3. Visualization
        t_vis_start = time.time()
        for track in online_tracks:
            track_id = track.track_id
            unique_track_ids.add(track_id)
            
            # Draw bounding box and label
            color = get_color(track_id)
            draw_sleek_box(frame, track.tlbr, track_id, color, class_names.get(track.class_id, "person"))
            
            # Draw fading trajectory tail
            draw_fade_tail(frame, track.history, color)
            
        # Render info panel (Sleek heads-up display overlay)
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (10, 10), (320, 120), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Print stats on frame
        fps_current = 1.0 / ((t_det_end - t_det_start) + (t_track_end - t_track_start) + 1e-6)
        cv2.putText(frame, f"THE AERIAL GUARDIAN PIPELINE", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {f_idx+1}/{num_frames}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Model: {model_name.split('/')[-1]}", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Active Tracks: {len(online_tracks)}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps_current:.1f}", (20, 110), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        t_vis_end = time.time()
        total_vis_time += (t_vis_end - t_vis_start)
        
        # Save frame to output video
        out_video.write(frame)
        
    if is_video:
        cap.release()
    out_video.release()
    
    # Calculate global execution metrics
    total_frames = processed_frames if processed_frames > 0 else 1
    avg_det_time = total_det_time / total_frames
    avg_track_time = total_track_time / total_frames
    avg_vis_time = total_vis_time / total_frames
    avg_total_time = avg_det_time + avg_track_time + avg_vis_time
    pipeline_fps = 1.0 / avg_total_time
    
    print("\n" + "="*50)
    print(" PIPELINE EXECUTION PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Processed Frames:       {total_frames}")
    print(f"Total Unique Targets:   {len(unique_track_ids)}")
    print(f"Avg Detection Time:     {avg_det_time*1000:.1f} ms")
    print(f"Avg Tracking (GMC):     {avg_track_time*1000:.1f} ms")
    print(f"Avg Rendering Time:     {avg_vis_time*1000:.1f} ms")
    print(f"Avg Total/Frame Time:   {avg_total_time*1000:.1f} ms")
    print(f"Overall Pipeline FPS:   {pipeline_fps:.2f}")
    print(f"Processed video saved to: {output_path}")
    print("="*50 + "\n")
    
    return pipeline_fps, len(unique_track_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial Guardian Detection and Tracking Pipeline")
    parser.add_argument("--sequence", type=str, required=True, help="Path to sequence directory containing frames")
    parser.add_argument("--model", type=str, default="mshamrai/yolov8s-visdrone", help="YOLO model path or HF identifier")
    parser.add_argument("--imgsz", type=int, default=1080, help="Inference resolution width")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--track-buffer", type=int, default=30, help="Frames to keep lost tracks")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="IoU distance matching threshold")
    parser.add_argument("--output", type=str, default="output/processed_video.mp4", help="Path to save output video")
    parser.add_argument("--limit-frames", type=int, default=-1, help="Limit number of frames to process (-1 for no limit)")
    
    args = parser.parse_args()
    
    tracker_config = {
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh
    }
    
    process_sequence(
        seq_dir=args.sequence,
        model_name=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        tracker_config=tracker_config,
        output_path=args.output,
        limit_frames=args.limit_frames
    )
