import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing

# --- 1. The Precomputed Dataset Loader (Static Shapes) ---
class PrecomputedFrequencyDataset(Dataset):
    def __init__(self, dataframe, max_patches=60):
        self.dataframe = dataframe
        self.max_patches = max_patches 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tensor_path = self.dataframe.iloc[idx]['tensor_path']
        label = self.dataframe.iloc[idx]['label']
        spectra = torch.load(tensor_path) 
        
        # Static shape padding for consistent CPU batching
        n_patches = spectra.shape[0]
        if n_patches < self.max_patches:
            padding = torch.zeros(self.max_patches - n_patches, 1, 256, 256)
            spectra = torch.cat([spectra, padding], dim=0)
        elif n_patches > self.max_patches:
            spectra = spectra[:self.max_patches]
            
        return spectra, torch.tensor(label, dtype=torch.float32)

# --- 2. The Streamlined Architecture ---
class FastFrequencyModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, stride=original_conv1.stride, padding=original_conv1.padding, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(torch.sum(original_conv1.weight, dim=1, keepdim=True))
                
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1) 
        )

    def forward(self, spectra):
        B, N, C, H, W = spectra.shape
        
        # 1. Flatten 5D to 4D -> [Batch * Patches, Channels, Height, Width]
        spectra = spectra.view(B * N, C, H, W) 
        
        # 2. NOW apply the CPU memory optimization since it is Rank 4
        spectra = spectra.to(memory_format=torch.channels_last)
        
        # 3. Pass through the backbone
        patch_logits = self.backbone(spectra) 
        
        # 4. Reshape and pool
        patch_logits = patch_logits.view(B, N, 1) 
        frame_logit, _ = torch.max(patch_logits, dim=1) 
        
        return frame_logit

# --- 3. Data Preparation & Splitting ---
def prepare_data(csv_path, batch_size=8): # Lower batch size is usually better for CPU cache
    df = pd.read_csv(csv_path)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['video_id']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Dataset split: {len(train_df)} Train frames | {len(val_df)} Val frames")

    train_dataset = PrecomputedFrequencyDataset(train_df)
    val_dataset = PrecomputedFrequencyDataset(val_df)

    # CPU Opt: pin_memory=False (no GPU transfer), optimal workers
    workers = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False, drop_last=True)
    
    return train_loader, val_loader

# --- 4. The CPU-Optimized Training Loop ---
def train_model(model, train_loader, val_loader, epochs=30, learning_rate=1e-4, save_path="best_cpu_model.pth"):
    device = torch.device("cpu")
    
    # CPU Opt: Prevent PyTorch from thrashing all logical cores
    torch.set_num_threads(multiprocessing.cpu_count()) 
    print(f"\n[INIT] Training on CPU with {torch.get_num_threads()} threads.")
    
    # CPU Opt: Convert model to Channels Last memory format
    model = model.to(memory_format=torch.channels_last)
    
    # CPU Opt: JIT compile the model (requires PyTorch 2.0+)
    try:
        print("[INIT] Compiling model with torch.compile() for CPU acceleration...")
        #model = torch.compile(model)
    except Exception as e:
        print(f"[INIT] torch.compile() not available or failed, proceeding with eager mode. ({e})")
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, leave=True)
        
        for spectra, labels in loop:
            # CPU Opt: Ensure incoming tensors match the channels_last format
            
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            
            # CPU Opt: BFloat16 Autocast for massive speedup on modern CPUs
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                frame_logits = model(spectra) 
                loss = criterion(frame_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * spectra.size(0)
            
            # Detach and convert to float32 for metric calculations
            preds = torch.round(torch.sigmoid(frame_logits.detach().float()))
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(acc=correct_train/total_train)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for spectra, labels in val_loader:
                
                labels = labels.unsqueeze(1)
                
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                    frame_logits = model(spectra)
                    loss = criterion(frame_logits, labels)
                
                val_loss += loss.item() * spectra.size(0)
                
                preds = torch.round(torch.sigmoid(frame_logits.float()))
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        avg_train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            print(f"[*] Val Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            # Un-compile before saving to ensure compatibility when loading elsewhere
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), save_path)
        else:
            print(f"[-] Val loss did not improve from {best_val_loss:.4f}.")
        
        print("") 
        scheduler.step(avg_val_loss)

    return history

# --- 5. Graph Generation ---
def plot_metrics(history, output_path="training_metrics_cpu.png"):
    print("Generating training metric graphs...")
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    CSV_PATH = "./dataset/tensor_labels.csv"
    
    # Dropped batch size to 8. CPUs have smaller L3 caches, so massive batches actually slow them down.
    train_dl, val_dl = prepare_data(CSV_PATH, batch_size=8) 
    model = FastFrequencyModel(pretrained=True)
    
    training_history = train_model(model, train_dl, val_dl, epochs=30, save_path="best_cpu_model.pth")
    plot_metrics(training_history, output_path="training_metrics_cpu.png")