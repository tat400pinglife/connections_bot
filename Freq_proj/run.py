import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

def calculate_radial_profile(data, center=None):
    """Calculates the 1D radial average of a 2D spectrum."""
    y, x = np.indices((data.shape))
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def analyze_and_visualize(tensor_dir="./dataset/tensors", output_dir="./dataset/visualizations"):
    print(f"\n[INIT] Booting Tensor Analytics Pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    
    pt_files = glob(os.path.join(tensor_dir, "*.pt"))
    if not pt_files:
        print(f"[!] No tensor files found in {tensor_dir}.")
        return

    # Assuming patches are 256x256 based on our architecture
    grid_size = 224
    
    # Use float64 for accumulators to prevent precision loss over thousands of files
    real_sum = torch.zeros((grid_size, grid_size), dtype=torch.float64)
    ai_sum = torch.zeros((grid_size, grid_size), dtype=torch.float64)
    
    real_count = 0
    ai_count = 0

    print(f"[*] Aggregating data from {len(pt_files)} tensors...")
    
    for file in tqdm(pt_files, desc="Processing Tensors"):
        try:
            # Load the self-contained payload we saved earlier
            payload = torch.load(file, map_location='cpu')
            tensor = payload['tensor'] # Shape: [60, 1, 256, 256]
            label = payload['label']   # 0 = Real, 1 = AI
            
            # The tensor contains multiple patches per frame. 
            # We average all patches to get a single 256x256 representation for this frame.
            frame_avg = tensor.mean(dim=0).squeeze() # Shape: [256, 256]
            
            if label == 0:
                real_sum += frame_avg.double()
                real_count += 1
            else:
                ai_sum += frame_avg.double()
                ai_count += 1
                
        except Exception as e:
            print(f"\n[!] Corrupted file skipped: {file} - {e}")

    if real_count == 0 or ai_count == 0:
        print("[!] Missing data for one of the categories. Ensure both Real and AI tensors exist.")
        return

    # Calculate final averages
    real_avg_spectrum = (real_sum / real_count).numpy()
    ai_avg_spectrum = (ai_sum / ai_count).numpy()
    
    # The Difference Map highlights the exact geometric divergence
    difference_map = ai_avg_spectrum - real_avg_spectrum

    # =====================================================================
    # GRAPH 1: 2D SPECTRAL HEATMAPS
    # =====================================================================
    print("\n[*] Generating 2D Spectral Heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Average FFT Spectral Density: Real vs. AI Video', fontsize=18, fontweight='bold')

    # Global min/max for consistent color scaling across graphs
    vmin = min(real_avg_spectrum.min(), ai_avg_spectrum.min())
    vmax = max(real_avg_spectrum.max(), ai_avg_spectrum.max())

    im0 = axes[0].imshow(real_avg_spectrum, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Real Video Spectrum (n={real_count})', fontsize=14)
    axes[0].axis('off')

    im1 = axes[1].imshow(ai_avg_spectrum, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'AI Video Spectrum (n={ai_count})', fontsize=14)
    axes[1].axis('off')

    # Difference map uses a diverging colormap (blue = negative, red = positive difference)
    im2 = axes[2].imshow(difference_map, cmap='coolwarm')
    axes[2].set_title('Absolute Difference Map (AI - Real)', fontsize=14)
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2D_Spectral_Comparison.png"), dpi=300)
    plt.close()

    # =====================================================================
    # GRAPH 2: 1D RADIAL PROFILE
    # =====================================================================
    print("[*] Generating 1D Radial Profile Plot...")
    real_radial = calculate_radial_profile(real_avg_spectrum)
    ai_radial = calculate_radial_profile(ai_avg_spectrum)

    plt.figure(figsize=(10, 6))
    plt.plot(real_radial, label='Real Videos', color='blue', linewidth=2)
    plt.plot(ai_radial, label='AI Generated Videos', color='red', linewidth=2)
    
    plt.title('1D Radial Power Spectrum', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency Radius (Low Freq -> High Freq)', fontsize=12)
    plt.ylabel('Log Amplitude', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Fill the area between curves to highlight the magnitude of the divergence
    plt.fill_between(range(len(real_radial)), real_radial, ai_radial, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1D_Radial_Profile.png"), dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Visualizations saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    analyze_and_visualize()