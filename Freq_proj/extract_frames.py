import os
import cv2
import csv
from pathlib import Path

def extract_frames(video_path, output_dir, label, video_id, frames_per_second=2):
    """
    Extracts frames from a video at a specified FPS and saves them losslessly.
    Returns a list of dictionaries containing frame metadata for the CSV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30 # Fallback if metadata is missing
        
    frame_interval = int(round(fps / frames_per_second))
    
    frame_count = 0
    saved_count = 0
    csv_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only extract frames at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_id}_frame_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save as Lossless PNG
            # IMWRITE_PNG_COMPRESSION ranges from 0 to 9. Lower is faster/larger size. 
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            csv_rows.append({
                'image_path': frame_path,
                'label': label,
                'video_id': video_id
            })
            saved_count += 1

        frame_count += 1

    cap.release()
    return csv_rows

def build_dataset(root_video_dir, output_root_dir, csv_output_path):
    os.makedirs(output_root_dir, exist_ok=True)
    
    categories = {'real': 0, 'ai': 1}
    all_dataset_rows = []
    
    for category, label in categories.items():
        video_dir = os.path.join(root_video_dir, category)
        if not os.path.exists(video_dir):
            print(f"Warning: Directory {video_dir} not found. Skipping.")
            continue
            
        print(f"\nProcessing '{category}' videos (Label: {label})...")
        
        for video_filename in os.listdir(video_dir):
            if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
                
            video_path = os.path.join(video_dir, video_filename)
            video_id = Path(video_filename).stem # Gets filename without extension
            
            print(f"  Extracting -> {video_filename}")
            rows = extract_frames(video_path, output_root_dir, label, video_id)
            all_dataset_rows.append(rows)

    flat_rows = [item for sublist in all_dataset_rows for item in sublist]

    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image_path', 'label', 'video_id'])
        writer.writeheader()
        writer.writerows(flat_rows)
        
    print(f"\nExtraction complete! {len(flat_rows)} total frames saved.")
    print(f"Dataset index saved to: {csv_output_path}")

if __name__ == "__main__":
    ROOT_VIDEOS = "./Video Folder" 
    EXTRACTED_FRAMES_DIR = "./dataset/frames"
    CSV_INDEX = "./dataset/labels.csv"
    
    build_dataset(ROOT_VIDEOS, EXTRACTED_FRAMES_DIR, CSV_INDEX)