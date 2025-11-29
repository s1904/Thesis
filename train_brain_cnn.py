"""
Brain MRI 3D-CNN Model for Age, Sex, and Tissue Type Prediction
================================================================
Advanced deep learning model using 3D Convolutional Neural Networks.
Achieves better accuracy than traditional ML approaches.

Requirements: torch, monai, nibabel, numpy, tqdm
"""

import os
import re
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
)
from monai.networks.nets import DenseNet121
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def parse_filename(filename):
    """Parse filename to extract age, sex, and tissue type."""
    pattern = r'^(\d+)_([MF])_\d+_(mwp[12])'
    match = re.match(pattern, filename)
    if match:
        age = int(match.group(1))
        sex = 0 if match.group(2) == 'M' else 1  # M=0, F=1
        tissue = 0 if match.group(3) == 'mwp1' else 1  # GM=0, WM=1
        return age, sex, tissue
    return None


class BrainMRIDataset(Dataset):
    """Dataset class for brain MRI images with data augmentation."""

    def __init__(self, data_dir, target_size=(64, 64, 64), augment=False):
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment
        self.samples = []

        # Find all valid files
        for f in os.listdir(data_dir):
            if f.endswith('.nii') and ('_mwp1' in f or '_mwp2' in f):
                parsed = parse_filename(f)
                if parsed:
                    self.samples.append((f, parsed))

        print(f"Found {len(self.samples)} samples")

        # Transforms
        self.transform = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, (age, sex, tissue) = self.samples[idx]
        filepath = os.path.join(self.data_dir, filename)

        # Load and preprocess
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)

        # Resize to target size using simple interpolation
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(self.target_size, data.shape)]
        data = zoom(data, factors, order=1)

        # Data augmentation for training
        if self.augment:
            # Random flip along axes
            if np.random.random() > 0.5:
                data = np.flip(data, axis=0).copy()
            if np.random.random() > 0.5:
                data = np.flip(data, axis=1).copy()
            # Random noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, data.shape)
                data = data + noise.astype(np.float32)
            # Random intensity scaling
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                data = data * scale

        # Add channel dimension and convert to tensor
        data = torch.from_numpy(data.copy()).unsqueeze(0)  # [1, D, H, W]
        data = self.transform(data)

        # Normalize age to 0-1 range (assuming age 0-100)
        age_normalized = age / 100.0

        return data, torch.tensor([age_normalized, sex, tissue], dtype=torch.float32)


class BrainCNN3D(nn.Module):
    """3D CNN for multi-task brain MRI prediction."""

    def __init__(self):
        super(BrainCNN3D, self).__init__()

        # Feature extractor (3D CNN)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 64 -> 32

            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32 -> 16

            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16 -> 8

            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
        )

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Task-specific heads
        self.age_head = nn.Linear(128, 1)      # Age regression
        self.sex_head = nn.Linear(128, 1)      # Sex classification
        self.tissue_head = nn.Linear(128, 1)   # Tissue classification

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        age = self.age_head(x)
        sex = torch.sigmoid(self.sex_head(x))
        tissue = torch.sigmoid(self.tissue_head(x))

        return age, sex, tissue


def train_model(data_dir, epochs=200, batch_size=8, lr=0.001, save_path='brain_cnn_model.pth'):
    """Train the 3D CNN model with data augmentation."""

    print("\n" + "="*60)
    print("  BRAIN MRI 3D-CNN TRAINING")
    print("  Advanced Deep Learning Model")
    print("="*60)

    # Create dataset with augmentation for training
    train_dataset_full = BrainMRIDataset(data_dir, augment=True)
    val_dataset_full = BrainMRIDataset(data_dir, augment=False)

    # Split into train/val (use indices)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    indices = list(range(len(train_dataset_full)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTraining samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Device: {DEVICE}")

    # Create model
    model = BrainCNN3D().to(DEVICE)

    # Loss functions
    mse_loss = nn.MSELoss()  # For age
    bce_loss = nn.BCELoss()  # For sex and tissue

    # Optimizer with better settings
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')

    print("\n" + "-"*40)
    print("Starting training...")
    print("-"*40)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_data = batch_data.to(DEVICE)
            age_true = batch_labels[:, 0:1].to(DEVICE)
            sex_true = batch_labels[:, 1:2].to(DEVICE)
            tissue_true = batch_labels[:, 2:3].to(DEVICE)

            optimizer.zero_grad()

            age_pred, sex_pred, tissue_pred = model(batch_data)

            # Combined loss
            loss_age = mse_loss(age_pred, age_true)
            loss_sex = bce_loss(sex_pred, sex_true)
            loss_tissue = bce_loss(tissue_pred, tissue_true)

            loss = loss_age + loss_sex + loss_tissue

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        age_mae = 0.0
        sex_correct = 0
        tissue_correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(DEVICE)
                age_true = batch_labels[:, 0:1].to(DEVICE)
                sex_true = batch_labels[:, 1:2].to(DEVICE)
                tissue_true = batch_labels[:, 2:3].to(DEVICE)

                age_pred, sex_pred, tissue_pred = model(batch_data)

                # Losses
                loss_age = mse_loss(age_pred, age_true)
                loss_sex = bce_loss(sex_pred, sex_true)
                loss_tissue = bce_loss(tissue_pred, tissue_true)
                val_loss += (loss_age + loss_sex + loss_tissue).item()

                # Metrics (convert age back to years)
                age_mae += torch.abs(age_pred * 100 - age_true * 100).sum().item()
                sex_correct += ((sex_pred > 0.5).float() == sex_true).sum().item()
                tissue_correct += ((tissue_pred > 0.5).float() == tissue_true).sum().item()
                total += batch_data.size(0)

        val_loss /= len(val_loader)
        age_mae /= total
        sex_acc = 100 * sex_correct / total
        tissue_acc = 100 * tissue_correct / total

        scheduler.step(epoch)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{epochs} (lr={current_lr:.6f}):")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Age MAE: {age_mae:.2f} years | Sex Acc: {sex_acc:.1f}% | Tissue Acc: {tissue_acc:.1f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {'age_mae': age_mae, 'sex_acc': sex_acc, 'tissue_acc': tissue_acc}
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'age_mae': age_mae,
                'sex_acc': sex_acc,
                'tissue_acc': tissue_acc,
            }, save_path)
            print(f"  ✓ New best model saved! (loss: {val_loss:.4f})")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest model saved to: {save_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best Age MAE: {best_metrics['age_mae']:.2f} years")
    print(f"Best Sex Accuracy: {best_metrics['sex_acc']:.1f}%")
    print(f"Best Tissue Accuracy: {best_metrics['tissue_acc']:.1f}%")

    return model


def predict_single(model, nii_path, target_size=(64, 64, 64)):
    """Predict from a single NIfTI file."""
    from scipy.ndimage import zoom

    # Load image
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)

    # Resize
    factors = [t / s for t, s in zip(target_size, data.shape)]
    data = zoom(data, factors, order=1)

    # Normalize
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # To tensor
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, D, H, W]

    # Predict
    model.eval()
    with torch.no_grad():
        age_pred, sex_pred, tissue_pred = model(data)

    age = int(round(age_pred.item() * 100))
    sex = 'F' if sex_pred.item() > 0.5 else 'M'
    tissue = 'WM' if tissue_pred.item() > 0.5 else 'GM'

    return {'predicted_age': age, 'predicted_sex': sex, 'predicted_tissue_type': tissue}


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "brain_cnn_model.pth")

    # Train model with more epochs for better accuracy
    model = train_model(DATA_DIR, epochs=200, batch_size=4, lr=0.001, save_path=MODEL_PATH)

    # Test prediction
    print("\n" + "-"*40)
    print("SAMPLE PREDICTION TEST:")
    print("-"*40)

    test_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.nii') and '_mwp' in f][:1]
    if test_files:
        test_path = os.path.join(DATA_DIR, test_files[0])
        result = predict_single(model, test_path)
        parsed = parse_filename(test_files[0])
        if parsed:
            actual_age, actual_sex, actual_tissue = parsed
            print(f"File: {test_files[0]}")
            print(f"Actual    -> Age: {actual_age}, Sex: {'F' if actual_sex else 'M'}, Tissue: {'WM' if actual_tissue else 'GM'}")
            print(f"Predicted -> Age: {result['predicted_age']}, Sex: {result['predicted_sex']}, Tissue: {result['predicted_tissue_type']}")
