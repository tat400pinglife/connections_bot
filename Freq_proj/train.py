import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- 1. The Precomputed Dataset Loader ---
class PrecomputedFrequencyDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tensor_path = self.dataframe.iloc[idx]['tensor_path']
        label = self.dataframe.iloc[idx]['label']
        spectra = torch.load(tensor_path) 
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
        spectra = spectra.squeeze(0) 
        patch_logits = self.backbone(spectra) 
        frame_logit, _ = torch.max(patch_logits, dim=0, keepdim=True) 
        return frame_logit

# --- 3. Data Preparation & Splitting ---
def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['video_id']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Dataset split: {len(train_df)} Train frames | {len(val_df)} Val frames")

    train_dataset = PrecomputedFrequencyDataset(train_df)
    val_dataset = PrecomputedFrequencyDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

# --- 4. The Training Loop ---
def train_model(model, train_loader, val_loader, epochs=15, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    accumulation_steps = 8 
    
    # Initialize history tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        optimizer.zero_grad()
        loop = tqdm(train_loader, leave=True)
        
        for i, (spectra, labels) in enumerate(loop):
            spectra, labels = spectra.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            frame_logits = model(spectra) 
            loss = criterion(frame_logits, labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            
            preds = torch.round(torch.sigmoid(frame_logits))
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(acc=correct_train/total_train)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra, labels = spectra.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                frame_logits = model(spectra)
                loss = criterion(frame_logits, labels)
                val_loss += loss.item()
                
                preds = torch.round(torch.sigmoid(frame_logits))
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        # Calculate Epoch Metrics
        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        # Save to history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}\n")
        
        scheduler.step(avg_val_loss)

    return history

# --- 5. Graph Generation ---
def plot_metrics(history, output_path="training_metrics.png"):
    """Generates and saves the loss and accuracy graphs."""
    print("Generating training metric graphs...")
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss (Binary Cross Entropy)', fontsize=12)
    ax1.legend(fontsize=11)

    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Metrics graph saved successfully to: {output_path}")

# --- Execution ---
if __name__ == "__main__":
    CSV_PATH = "./dataset/tensor_labels.csv"
    
    # Make sure to install matplotlib: pip install matplotlib
    train_dl, val_dl = prepare_data(CSV_PATH)
    model = FastFrequencyModel(pretrained=True)
    
    # Run training and capture the returned history dictionary
    training_history = train_model(model, train_dl, val_dl, epochs=20)
    
    # Generate the graph
    plot_metrics(training_history, output_path="training_metrics.png")