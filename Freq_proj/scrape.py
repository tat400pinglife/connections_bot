import os
import time
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yt_dlp

warnings.filterwarnings("ignore")

# =====================================================================
# 1. YT-DLP DOWNLOADER
# =====================================================================
def download_video(url: str, path: Path, video_id: str, limit_10_sec: bool) -> Path:
    """Downloads the video using yt-dlp and returns the local path."""
    out_tmpl = os.path.join(str(path), f"{video_id}.%(ext)s")
    
    ydl_opts = {
        'format': 'best', 
        'match_filter': yt_dlp.utils.match_filter_func("!is_live"), 
        'outtmpl': out_tmpl,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }
    
    if limit_10_sec:
        ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(0, 10)])
        ydl_opts['force_keyframes_at_cuts'] = True
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    for f in os.listdir(path):
        if f.startswith(video_id) and f.endswith(('.mp4', '.webm', '.mkv')):
            return Path(os.path.join(path, f))
            
    raise Exception(f"Failed to locate downloaded file for {video_id}")

# =====================================================================
# 2. BATCH MANAGER
# =====================================================================
def download_batch(config: dict) -> None:
    csv_path = config["csv_path"]
    base_download_dir = config["raw_video_dir"]
    goal = config["batch_goal"]
    wait = config["delay_between_videos"]
    limit_10_sec = config["limit_to_10_seconds"]
    
    if not os.path.exists(csv_path):
        print(f"[!] No CSV found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    pending_df = df[df['status'] == 'pending']
    
    length = len(pending_df)
    if length == 0:
        print("[SUCCESS] Queue empty. No 'pending' links found.")
        return
    
    end = min(length, goal)
    videos_to_process = pending_df.iloc[:end]
    
    print(f"\n[INIT] Queue loaded. Downloading {end} videos...")
    
    for idx, row in tqdm(videos_to_process.iterrows(), total=end, desc="Downloading Batch"):
        video_id = row['video_id']
        url = row['url']
        category = row['category'] # e.g., 'real' or 'ai'
        
        # --- NEW: Dynamic Folder Routing ---
        category_dir = os.path.join(base_download_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        # -----------------------------------
        
        try:
            # Pass the specific category directory to the downloader
            download_video(url, category_dir, video_id, limit_10_sec)
            df.at[idx, 'status'] = 'completed'
        except Exception as e:
            print(f"\n[!] Error downloading {video_id}: {e}")
            df.at[idx, 'status'] = 'failed'
            
        df.to_csv(csv_path, index=False)
        time.sleep(wait)

    print(f"\n[COMPLETE] Finished processing batch of {end} videos.")

# =====================================================================
# CONFIGURATION & RUNNER
# =====================================================================
if __name__ == "__main__":
    
    CONFIG = {
        "csv_path": Path("./dataset/extraction_queue.csv"),
        "raw_video_dir": Path("./dataset/raw_videos"), # Base directory
        
        "batch_goal": 160,              
        "delay_between_videos": 3,     
        
        "limit_to_10_seconds": True    
    }
    
    os.makedirs(CONFIG['raw_video_dir'], exist_ok=True)
    
    download_batch(CONFIG)