import os
import pandas as pd
import torch
import torch.nn as nn
import torch.fft
import math
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --- 1. Core Architecture Modules ---

class SpectrumExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Convert to grayscale
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
            
        fft_out = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft_out, dim=(-2, -1))
        return torch.log(torch.abs(fft_shifted) + 1e-8)

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
        
        patches = x.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4) 
        return patches

# --- 2. The Conversion Pipeline ---

def precompute_pngs_to_tensors(input_csv, output_tensor_dir, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FFT precomputations on: {device}")

    os.makedirs(output_tensor_dir, exist_ok=True)

    # Initialize models and send to GPU
    patch_extractor = PatchExtractor(patch_size=256).to(device)
    fft_extractor = SpectrumExtractor().to(device)

    # Read the original PNG index
    df = pd.read_csv(input_csv)
    new_csv_rows = []

    # Loop through every PNG
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing PNGs"):
        img_path = row['image_path']
        label = row['label']
        video_id = row['video_id']

        if not os.path.exists(img_path):
            print(f"\nWarning: Missing file at {img_path}. Skipping.")
            continue

        # Load PNG, convert to tensor [1, C, H, W], normalize 0-1, send to GPU
        image = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. Slice into patches
            patches = patch_extractor(img_tensor)
            _, N, C, P_H, P_W = patches.shape
            
            # 2. Flatten and run FFT -> [N_Patches, 1, 256, 256]
            flat_patches = patches.view(N, C, P_H, P_W)
            spectra = fft_extractor(flat_patches)

        # Move tensor back to CPU and save
        tensor_filename = f"{video_id}_frame_{index:05d}.pt"
        tensor_path = os.path.join(output_tensor_dir, tensor_filename)
        
        torch.save(spectra.cpu(), tensor_path)

        # Log for the new CSV
        new_csv_rows.append({
            'tensor_path': tensor_path,
            'label': label,
            'video_id': video_id
        })

    # Save the new master index
    pd.DataFrame(new_csv_rows).to_csv(output_csv, index=False)
    print(f"\nSuccess! Saved {len(new_csv_rows)} tensor files.")
    print(f"New dataset index saved to: {output_csv}")

# --- Execution ---
if __name__ == "__main__":
    # Ensure these paths match your current setup
    INPUT_CSV = "./dataset/labels.csv"
    OUTPUT_TENSOR_DIR = "./dataset/tensors"
    OUTPUT_CSV = "./dataset/tensor_labels.csv"
    
    precompute_pngs_to_tensors(INPUT_CSV, OUTPUT_TENSOR_DIR, OUTPUT_CSV)