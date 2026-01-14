"""
Test Predictions on Test Set
=============================
Run predictions on all test files and calculate accuracy metrics.
"""

import os
import re
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import zoom
import pickle
from train_brain_model import BrainMRIPredictor, extract_features

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_filename(filename):
    """Parse filename to extract ground truth labels."""
    # Pattern 1: mwp1 or mwp2 files
    pattern1 = r'^(\d+)_([MF])_\d+_(mwp[12])'
    match = re.match(pattern1, filename)
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue = 'GM' if match.group(3) == 'mwp1' else 'WM'
        return age, sex, tissue

    # Pattern 2: m0wrp1s or m0wrp2s files
    pattern2 = r'^(\d+)_([MF])_(m0wrp[12]s)'
    match = re.match(pattern2, filename)
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue = 'GM' if 'm0wrp1s' in match.group(3) else 'WM'
        return age, sex, tissue

    return None


class BrainCNN3D(nn.Module):
    """Improved 3D CNN - must match training architecture."""
    def __init__(self):
        super(BrainCNN3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(0.1),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(0.1),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(0.2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(0.2),
            nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.age_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))
        self.sex_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))
        self.tissue_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        age = self.age_head(x)
        sex_logits = self.sex_head(x)
        tissue_logits = self.tissue_head(x)
        return age, torch.sigmoid(sex_logits), torch.sigmoid(tissue_logits)


def predict_cnn(model, nii_path, target_size=(64, 64, 64)):
    """Predict using CNN model."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)

        # Handle NaN and Inf values
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

        factors = [t / s for t, s in zip(target_size, data.shape)]
        data = zoom(data, factors, order=1)

        # Normalize
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        # Handle any remaining NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            age_pred, sex_pred, tissue_pred = model(tensor)

        # Handle NaN in predictions
        age_val = age_pred.item() * 100
        if np.isnan(age_val) or np.isinf(age_val):
            age_val = 0

        age = int(round(np.clip(age_val, 0, 100)))
        sex = 'F' if sex_pred.item() > 0.5 else 'M'
        tissue = 'WM' if tissue_pred.item() > 0.5 else 'GM'
        return age, sex, tissue
    except Exception as e:
        print(f"    ERROR in CNN prediction: {e}")
        return 0, 'M', 'GM'


def predict_sklearn(model, nii_path):
    """Predict using sklearn model."""
    try:
        features = extract_features(nii_path)
        if features is None:
            return 0, 'M', 'GM'
        age_pred, sex_pred, tissue_pred = model.predict(features.reshape(1, -1))
        age_val = age_pred[0]
        if np.isnan(age_val) or np.isinf(age_val):
            age_val = 0
        return int(round(np.clip(age_val, 0, 100))), sex_pred[0], tissue_pred[0]
    except Exception as e:
        print(f"    ERROR in Sklearn prediction: {e}")
        return 0, 'M', 'GM'


def test_all_files(test_dir, cnn_model_path, pkl_model_path):
    """Test all files and calculate accuracy."""
    print("\n" + "="*80)
    print("  TESTING PREDICTIONS ON TEST SET")
    print("="*80)
    
    # Load models
    print("\n📦 Loading models...")
    cnn_model = BrainCNN3D().to(DEVICE)
    checkpoint = torch.load(cnn_model_path, map_location=DEVICE, weights_only=False)
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ CNN model loaded (Age MAE: {checkpoint.get('age_mae', 'N/A'):.2f} years)")
    
    with open(pkl_model_path, 'rb') as f:
        sklearn_model = pickle.load(f)
    print(f"  ✓ Sklearn model loaded")
    
    # Get test files
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.nii')])
    print(f"\n📊 Testing on {len(test_files)} files...")
    
    results = []
    cnn_age_errors, sklearn_age_errors = [], []
    cnn_sex_correct, sklearn_sex_correct = 0, 0
    cnn_tissue_correct, sklearn_tissue_correct = 0, 0
    
    print("\n" + "-"*80)
    for i, filename in enumerate(test_files, 1):
        filepath = os.path.join(test_dir, filename)
        parsed = parse_filename(filename)
        
        if parsed is None:
            continue
        
        true_age, true_sex, true_tissue = parsed
        
        # CNN prediction
        cnn_age, cnn_sex, cnn_tissue = predict_cnn(cnn_model, filepath)
        
        # Sklearn prediction
        sklearn_age, sklearn_sex, sklearn_tissue = predict_sklearn(sklearn_model, filepath)
        
        # Calculate errors
        cnn_age_error = abs(cnn_age - true_age)
        sklearn_age_error = abs(sklearn_age - true_age)
        cnn_age_errors.append(cnn_age_error)
        sklearn_age_errors.append(sklearn_age_error)
        
        cnn_sex_correct += (cnn_sex == true_sex)
        sklearn_sex_correct += (sklearn_sex == true_sex)
        
        cnn_tissue_correct += (cnn_tissue == true_tissue)
        sklearn_tissue_correct += (sklearn_tissue == true_tissue)
        
        # Print result
        print(f"[{i:2d}] {filename[:50]:<50}")
        print(f"     TRUE:    Age={true_age:2d}  Sex={true_sex}  Tissue={true_tissue}")
        print(f"     CNN:     Age={cnn_age:2d} (±{cnn_age_error})  Sex={cnn_sex} {'✓' if cnn_sex==true_sex else '✗'}  Tissue={cnn_tissue} {'✓' if cnn_tissue==true_tissue else '✗'}")
        print(f"     SKLEARN: Age={sklearn_age:2d} (±{sklearn_age_error})  Sex={sklearn_sex} {'✓' if sklearn_sex==true_sex else '✗'}  Tissue={sklearn_tissue} {'✓' if sklearn_tissue==true_tissue else '✗'}")
        print()

    # Calculate overall accuracy
    total = len(test_files)
    cnn_age_mae = np.mean(cnn_age_errors)
    sklearn_age_mae = np.mean(sklearn_age_errors)
    cnn_sex_acc = 100 * cnn_sex_correct / total
    sklearn_sex_acc = 100 * sklearn_sex_correct / total
    cnn_tissue_acc = 100 * cnn_tissue_correct / total
    sklearn_tissue_acc = 100 * sklearn_tissue_correct / total

    print("="*80)
    print("  OVERALL ACCURACY RESULTS")
    print("="*80)
    print(f"\n{'Model':<15} {'Age MAE':<15} {'Sex Accuracy':<20} {'Tissue Accuracy':<20}")
    print("-"*80)
    print(f"{'CNN':<15} {cnn_age_mae:<15.2f} {cnn_sex_acc:<20.1f}% {cnn_tissue_acc:<20.1f}%")
    print(f"{'Sklearn':<15} {sklearn_age_mae:<15.2f} {sklearn_sex_acc:<20.1f}% {sklearn_tissue_acc:<20.1f}%")

    print("\n" + "="*80)
    print("  ACCURACY TARGETS: 85-95%")
    print("="*80)

    # Check if targets are met
    def check_target(value, name, model_name):
        if 85 <= value <= 95:
            status = "✓ PASS"
            color = "🟢"
        elif value > 95:
            status = "✓ EXCELLENT"
            color = "🟢"
        else:
            status = "✗ NEEDS IMPROVEMENT"
            color = "🔴"
        print(f"{color} {model_name} {name}: {value:.1f}% - {status}")

    print("\nCNN Model:")
    check_target(cnn_sex_acc, "Sex Accuracy", "CNN")
    check_target(cnn_tissue_acc, "Tissue Accuracy", "CNN")
    age_acc_cnn = 100 * (1 - cnn_age_mae / 100)  # Convert MAE to accuracy percentage
    print(f"  CNN Age MAE: {cnn_age_mae:.2f} years")

    print("\nSklearn Model:")
    check_target(sklearn_sex_acc, "Sex Accuracy", "Sklearn")
    check_target(sklearn_tissue_acc, "Tissue Accuracy", "Sklearn")
    print(f"  Sklearn Age MAE: {sklearn_age_mae:.2f} years")

    print("\n" + "="*80)


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(SCRIPT_DIR, "TestData")
    CNN_MODEL = os.path.join(SCRIPT_DIR, "brain_cnn_model.pth")
    PKL_MODEL = os.path.join(SCRIPT_DIR, "brain_mri_model.pkl")

    test_all_files(TEST_DIR, CNN_MODEL, PKL_MODEL)

