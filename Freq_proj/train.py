import warnings
warnings.filterwarnings("ignore")

import os
import glob
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch._dynamo
torch._dynamo.config.suppress_errors = True # Bypasses the missing cl.exe C++ compiler error

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SelfContainedFrequencyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the dictionary payload
        payload = torch.load(self.file_paths[idx])
        
        # Extract the embedded tensor and cast back to float32 for stable DataLoader collation
        spectra = payload['tensor'].float() 
        label = payload['label']
        
        return spectra, torch.tensor(label, dtype=torch.float32)

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
        spectra = spectra.view(B * N, C, H, W) 
        
        # Channel-last memory optimization
        spectra = spectra.to(memory_format=torch.channels_last)
        
        patch_logits = self.backbone(spectra) 
        patch_logits = patch_logits.view(B, N, 1) 
        frame_logit, _ = torch.max(patch_logits, dim=1) 
        
        return frame_logit

def prepare_data(tensor_dir, batch_size=16):
    all_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
    if not all_files:
        raise ValueError(f"No .pt files found in {tensor_dir}. Check your extraction path.")
    
    # Extract video_ids to prevent data leakage across Train/Val
    video_ids = [os.path.basename(f).split('_frame_')[0] for f in all_files]
    df = pd.DataFrame({'file_path': all_files, 'video_id': video_ids})
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['video_id']))
    
    train_files = df.iloc[train_idx]['file_path'].tolist()
    val_files = df.iloc[val_idx]['file_path'].tolist()
    
    print(f"Dataset split: {len(train_files)} Train frames | {len(val_files)} Val frames")

    train_dataset = SelfContainedFrequencyDataset(train_files)
    val_dataset = SelfContainedFrequencyDataset(val_files)

    # RAM Fix: num_workers=0 and pin_memory=False stops background process bloat
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=30, learning_rate=1e-4, save_path="best_cuda_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Training securely on: {device}")
        
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Automatic Mixed Precision (AMP) Scaler for speed
    scaler = torch.cuda.amp.GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training 
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, leave=True)
        
        for spectra, labels in loop:
            spectra = spectra.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True) 
            
            # AMP Forward Pass (16-bit math)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                frame_logits = model(spectra) 
                loss = criterion(frame_logits, labels)
            
            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * spectra.size(0)
            
            preds = torch.round(torch.sigmoid(frame_logits.detach().float()))
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(acc=correct_train/total_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra = spectra.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    frame_logits = model(spectra)
                    loss = criterion(frame_logits, labels)
                
                val_loss += loss.item() * spectra.size(0)
                
                preds = torch.round(torch.sigmoid(frame_logits.float()))
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        # Metrics/Save Checkpoint
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
            
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), save_path)
        else:
            print(f"[-] Val loss did not improve from {best_val_loss:.4f}.")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("") 
        scheduler.step(avg_val_loss)

    return history

def plot_metrics(history, output_path="training_metrics.png"):
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
    print(f"Metrics graph saved successfully to: {output_path}")

if __name__ == "__main__":
    TENSOR_DIR = "./dataset/tensors"
    
    # If CUDA OutOfMemory error, change batch_size to 8 or 4
    train_dl, val_dl = prepare_data(TENSOR_DIR, batch_size=16) 
    
    model = FastFrequencyModel(pretrained=True)
    
    # Start loop
    training_history = train_model(
        model=model, 
        train_loader=train_dl, 
        val_loader=val_dl, 
        epochs=30, 
        save_path="best_cuda_model.pth"
    )
    
    # Output the final training graphs
    plot_metrics(training_history, output_path="training_metrics.png")