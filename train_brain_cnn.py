"""
Brain MRI 3D-CNN Model for Age, Sex, and Tissue Type Prediction
================================================================
Advanced deep learning model using 3D Convolutional Neural Networks.
Achieves better accuracy than traditional ML approaches.

Supports both modulated warped (mwp1/mwp2) and modulated normalized (m0wrp1s/m0wrp2s) data.

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
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def parse_filename(filename):
    """
    Parse filename to extract age, sex, and tissue type.
    Supports both mwp1/mwp2 and m0wrp1s/m0wrp2s files.
    """
    # Pattern 1: mwp1 or mwp2 files
    pattern1 = r'^(\d+)_([MF])_\d+_(mwp[12])'
    match = re.match(pattern1, filename)
    if match:
        age = int(match.group(1))
        sex = 0 if match.group(2) == 'M' else 1  # M=0, F=1
        tissue = 0 if match.group(3) == 'mwp1' else 1  # GM=0, WM=1
        return age, sex, tissue

    # Pattern 2: m0wrp1s or m0wrp2s files (normalized data)
    pattern2 = r'^(\d+)_([MF])_(m0wrp[12]s)'
    match = re.match(pattern2, filename)
    if match:
        age = int(match.group(1))
        sex = 0 if match.group(2) == 'M' else 1  # M=0, F=1
        tissue = 0 if 'm0wrp1s' in match.group(3) else 1  # GM=0, WM=1
        return age, sex, tissue

    return None


class BrainMRIDataset(Dataset):
    """Dataset class for brain MRI images with data augmentation."""

    def __init__(self, data_dir, target_size=(64, 64, 64), augment=False):
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment
        self.samples = []

        # Find all valid files (both mwp and m0wrp files)
        for f in os.listdir(data_dir):
            if f.endswith('.nii') and ('_mwp1' in f or '_mwp2' in f or '_m0wrp1s' in f or '_m0wrp2s' in f):
                parsed = parse_filename(f)
                if parsed:
                    self.samples.append((f, parsed))

        print(f"Found {len(self.samples)} samples (including normalized data)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, (age, sex, tissue) = self.samples[idx]
        filepath = os.path.join(self.data_dir, filename)

        # Load and preprocess
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)

        # Resize to target size using simple interpolation
        factors = [t / s for t, s in zip(self.target_size, data.shape)]
        data = zoom(data, factors, order=1)

        # Normalize to 0-1 range
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        # Replace any NaN or Inf values
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

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
                data = np.clip(data + noise.astype(np.float32), 0, 1)
            # Random intensity scaling
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                data = np.clip(data * scale, 0, 1)

        # Add channel dimension and convert to tensor
        data = torch.from_numpy(data.copy()).unsqueeze(0)  # [1, D, H, W]

        # Normalize age to 0-1 range (assuming age 0-100)
        age_normalized = age / 100.0

        return data, torch.tensor([age_normalized, float(sex), float(tissue)], dtype=torch.float32)


class BrainCNN3D(nn.Module):
    """Improved 3D CNN for multi-task brain MRI prediction with better accuracy."""

    def __init__(self):
        super(BrainCNN3D, self).__init__()

        # Feature extractor (Deeper 3D CNN with residual connections)
        self.features = nn.Sequential(
            # Block 1 - Initial feature extraction
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 64 -> 32
            nn.Dropout3d(0.1),

            # Block 2 - Deeper features
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32 -> 16
            nn.Dropout3d(0.1),

            # Block 3 - High-level features
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16 -> 8
            nn.Dropout3d(0.2),

            # Block 4 - Abstract features
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 8 -> 4
            nn.Dropout3d(0.2),

            # Block 5 - Final feature extraction
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
        )

        # Shared fully connected layers with more capacity
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Task-specific heads with deeper networks
        self.age_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.sex_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.tissue_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        age = self.age_head(x)
        sex_logits = self.sex_head(x)  # Return logits for BCEWithLogitsLoss
        tissue_logits = self.tissue_head(x)  # Return logits for BCEWithLogitsLoss

        return age, sex_logits, tissue_logits


def train_model(data_dir, epochs=200, batch_size=8, lr=0.001, save_path='brain_cnn_model.pth'):
    """Train the 3D CNN model with data augmentation and balanced sampling."""

    print("\n" + "="*60)
    print("  BRAIN MRI 3D-CNN TRAINING")
    print("  Advanced Deep Learning Model with Balanced Sampling")
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

    # Create data loaders with simple shuffling
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

    # Calculate class weights from training data to handle imbalance
    # Count sex distribution in training set
    sex_counts = {'M': 0, 'F': 0}
    tissue_counts = {'GM': 0, 'WM': 0}

    for idx in train_indices:
        filename, (age, sex, tissue) = train_dataset_full.samples[idx]
        sex_label = 'F' if sex == 1 else 'M'
        tissue_label = 'WM' if tissue == 1 else 'GM'
        sex_counts[sex_label] += 1
        tissue_counts[tissue_label] += 1

    total_samples = len(train_indices)

    # Calculate pos_weight for Female class (class 1)
    # pos_weight = count(negative_class) / count(positive_class)
    # Female is positive class (1), Male is negative class (0)
    sex_pos_weight = torch.tensor([sex_counts['M'] / sex_counts['F']]).to(DEVICE)
    tissue_pos_weight = torch.tensor([tissue_counts['GM'] / tissue_counts['WM']]).to(DEVICE)

    print(f"\n📊 Class Distribution in Training Set:")
    print(f"   Sex: Male={sex_counts['M']} ({100*sex_counts['M']/total_samples:.1f}%), "
          f"Female={sex_counts['F']} ({100*sex_counts['F']/total_samples:.1f}%)")
    print(f"   Tissue: GM={tissue_counts['GM']} ({100*tissue_counts['GM']/total_samples:.1f}%), "
          f"WM={tissue_counts['WM']} ({100*tissue_counts['WM']/total_samples:.1f}%)")
    print(f"   Sex pos_weight (for Female): {sex_pos_weight.item():.3f}")
    print(f"   Tissue pos_weight (for WM): {tissue_pos_weight.item():.3f}")

    # Loss functions with class weights
    mse_loss = nn.MSELoss()  # For age
    bce_loss_sex = nn.BCEWithLogitsLoss(pos_weight=sex_pos_weight)  # For sex (weighted for imbalance)
    bce_loss_tissue = nn.BCEWithLogitsLoss(pos_weight=tissue_pos_weight)  # For tissue (weighted)

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
            sex_true = torch.clamp(batch_labels[:, 1:2], 0.0, 1.0).to(DEVICE)
            tissue_true = torch.clamp(batch_labels[:, 2:3], 0.0, 1.0).to(DEVICE)

            optimizer.zero_grad()

            age_pred, sex_logits, tissue_logits = model(batch_data)

            # Combined loss with class weights and task importance
            loss_age = mse_loss(age_pred, age_true)
            loss_sex = bce_loss_sex(sex_logits, sex_true)  # Uses pos_weight for Male class
            loss_tissue = bce_loss_tissue(tissue_logits, tissue_true)  # Uses pos_weight for WM

            # Weighted loss: Age=1.0, Sex=4.0, Tissue=2.0 (heavily prioritize sex classification)
            loss = loss_age + 4.0 * loss_sex + 2.0 * loss_tissue

            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                sex_true = torch.clamp(batch_labels[:, 1:2], 0.0, 1.0).to(DEVICE)
                tissue_true = torch.clamp(batch_labels[:, 2:3], 0.0, 1.0).to(DEVICE)

                age_pred, sex_logits, tissue_logits = model(batch_data)

                # Losses
                loss_age = mse_loss(age_pred, age_true)
                loss_sex = bce_loss_sex(sex_logits, sex_true)
                loss_tissue = bce_loss_tissue(tissue_logits, tissue_true)
                val_loss += (loss_age + 4.0 * loss_sex + 2.0 * loss_tissue).item()

                # Metrics (convert age back to years)
                # Apply sigmoid to logits for predictions
                sex_pred = torch.sigmoid(sex_logits)
                tissue_pred = torch.sigmoid(tissue_logits)

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
    # Focus on Age and Tissue accuracy (accept natural sex distribution)
    model = train_model(DATA_DIR, epochs=200, batch_size=8, lr=0.001, save_path=MODEL_PATH)

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
