import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import torch
import torch.nn as nn
import torch.fft
import math
import gc 
import time 
from pathlib import Path

class SpectrumExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Convert RGB to Grayscale dynamically
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
            
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return torch.log(torch.abs(x) + 1e-8)

class PatchExtractor(nn.Module):
    def __init__(self, patch_size=256):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = max(0, self.patch_size - H)
        pad_w = max(0, self.patch_size - W)
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2:]

        stride_h = self.patch_size if H % self.patch_size == 0 else math.floor((H - self.patch_size) / math.ceil(H / self.patch_size - 1))
        stride_w = self.patch_size if W % self.patch_size == 0 else math.floor((W - self.patch_size) / math.ceil(W / self.patch_size - 1))
        
        x = x.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4) 
        return x

def process_and_save_frame(numpy_frame, patch_extractor, fft_extractor, device, config, label, video_id, save_path):
    """Everything inside this function dies the moment it returns, preventing RAM leaks."""
    with torch.no_grad(): 
        # Convert numpy to tensor and move to GPU
        frame_t = cv2.cvtColor(numpy_frame, cv2.COLOR_BGR2RGB)
        frame_t = torch.from_numpy(frame_t).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        
        # Slicing and Math Pipeline
        frame_t = patch_extractor(frame_t)
        _, N, C, P_H, P_W = frame_t.shape
        frame_t = frame_t.view(N, C, P_H, P_W)
        frame_t = fft_extractor(frame_t)
        
        # Static Shape Padding Logic
        n_patches = frame_t.shape[0]
        if n_patches < config["max_patches"]:
            padding = torch.zeros(config["max_patches"] - n_patches, 1, config["patch_size"], config["patch_size"], device=device)
            frame_t = torch.cat([frame_t, padding], dim=0)
        elif n_patches > config["max_patches"]:
            frame_t = frame_t[:config["max_patches"]]
        
        # .half() compresses the 32-bit floats to 16-bit floats right before saving to disk
        final_tensor = frame_t.detach().cpu().half().clone()
        
        # Build and save self-contained payload
        tensor_payload = {
            'tensor': final_tensor, 
            'label': label,         
            'video_id': video_id    
        }
        torch.save(tensor_payload, save_path)


def extract_tensors(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Running compression-optimized extraction on: {device}")
    print(f"[INIT] Saving files to: {config['output_tensors']}\n")

    os.makedirs(os.path.normpath(config['output_tensors']), exist_ok=True)

    # Initialize models with the config settings
    patch_extractor = PatchExtractor(patch_size=config["patch_size"]).to(device)
    fft_extractor = SpectrumExtractor().to(device)
    
    categories = {'real': 0, 'ai': 1}

    for category, label in categories.items():
        video_dir = os.path.join(config["root_videos"], category)
        if not os.path.exists(video_dir):
            print(f"Warning: Directory '{video_dir}' not found. Skipping.")
            continue
            
        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"\n--- Starting '{category}' category ({len(video_files)} videos) ---")
        
        for idx, video_filename in enumerate(video_files):
            print(f"[{idx+1}/{len(video_files)}] Processing: {video_filename}")
            
            video_path = os.path.join(video_dir, video_filename)
            video_id = Path(video_filename).stem
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened(): 
                print(f"  -> OpenCV failed to open {video_filename}. Skipping.")
                continue

            fps_meta = cap.get(cv2.CAP_PROP_FPS)
            fps_meta = fps_meta if (fps_meta and fps_meta > 0) else 30 
            frame_interval = int(round(fps_meta / config["frames_per_second"]))
            
            frame_count = 0
            saved_count = 0
            start_time = time.time()
            
            while True:
                # The Timeout Trap
                if time.time() - start_time > config["timeout_seconds"]:
                    print(f"  -> [!] TIMEOUT: {video_filename} took over {config['timeout_seconds']}s. Abandoning.")
                    break

                ret, frame = cap.read()
                if not ret: 
                    break

                if frame_count % frame_interval == 0:
                    tensor_filename = f"{video_id}_frame_{saved_count:04d}.pt"
                    tensor_path = os.path.join(config["output_tensors"], tensor_filename)
                    
                    # Pass the raw frame and the entire config to the isolated function
                    process_and_save_frame(
                        numpy_frame=frame, 
                        patch_extractor=patch_extractor, 
                        fft_extractor=fft_extractor, 
                        device=device, 
                        config=config, 
                        label=label, 
                        video_id=video_id, 
                        save_path=tensor_path
                    )
                    
                    saved_count += 1
                    
                    # Optional: Force garbage collection mid-video for extremely long clips
                    if saved_count % 50 == 0:
                        gc.collect()
                    
                frame_count += 1
                
                # Failsafe limit
                if saved_count >= config["max_frames_per_video"]:
                    print(f"  -> [!] WARNING: Hit {config['max_frames_per_video']} frame cap on {video_filename}. Breaking loop.")
                    break
                    
            cap.release()
            
            # Clear CUDA cache between videos just to be safe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    
    CONFIG = {
        # Directories
        "root_videos": "./dataset/raw_videos",
        "output_tensors": "./dataset/tensors", 
        
        # Sampling Strategy
        "frames_per_second": 2,           # How many frames to pull out of every second of video
        "max_frames_per_video": 300,      # Absolute cutoff. Prevents infinite looping on broken files.
        "timeout_seconds": 30,            # Max time to spend on a single video before skipping
        
        # Mathematical Architecture
        "patch_size": 224,                # Dimensions of the grid squares (
        
        # Hardware & Storage Constraints
        "max_patches": 30                 # Max grid squares kept per frame. 
    }
    
    extract_tensors(CONFIG)