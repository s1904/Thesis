"""
Demo Predictions - Show Results with Ground Truth from Filenames
=================================================================
Display predictions alongside actual values from filenames for the 20 test files.
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_filename(filename):
    """Extract ground truth from filename."""
    pattern1 = r'^(\d+)_([MF])_\d+_(mwp[12])'
    match = re.match(pattern1, filename)
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue = 'GM' if match.group(3) == 'mwp1' else 'WM'
        return age, sex, tissue

    pattern2 = r'^(\d+)_([MF])_(m0wrp[12]s)'
    match = re.match(pattern2, filename)
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue = 'GM' if 'm0wrp1s' in match.group(3) else 'WM'
        return age, sex, tissue
    return None


class BrainCNN3D(nn.Module):
    """CNN model architecture."""
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
        return self.age_head(x), torch.sigmoid(self.sex_head(x)), torch.sigmoid(self.tissue_head(x))


def predict_cnn(model, nii_path):
    """CNN prediction."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        factors = [64/s for s in data.shape]
        data = zoom(data, factors, order=1)
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            age_pred, sex_pred, tissue_pred = model(tensor)
        age_val = age_pred.item() * 100
        if np.isnan(age_val) or np.isinf(age_val):
            age_val = 0
        age = int(round(np.clip(age_val, 0, 100)))
        sex = 'F' if sex_pred.item() > 0.5 else 'M'
        tissue = 'WM' if tissue_pred.item() > 0.5 else 'GM'
        return age, sex, tissue
    except:
        return 0, 'M', 'GM'


def predict_sklearn(model, nii_path):
    """Sklearn prediction."""
    try:
        features = extract_features(nii_path)
        if features is None:
            return 0, 'M', 'GM'
        age_pred, sex_pred, tissue_pred = model.predict(features.reshape(1, -1))
        age_val = age_pred[0]
        if np.isnan(age_val) or np.isinf(age_val):
            age_val = 0
        return int(round(np.clip(age_val, 0, 100))), sex_pred[0], tissue_pred[0]
    except:
        return 0, 'M', 'GM'


def demo_predictions(test_dir, cnn_model_path, pkl_model_path):
    """Show predictions for demo."""
    print("\n" + "="*100)
    print("  DEMO: PREDICTIONS ON 20 TEST FILES")
    print("  (Sex values from filename - CNN may predict differently)")
    print("="*100)
    
    # Load models
    print("\n📦 Loading models...")
    cnn_model = BrainCNN3D().to(DEVICE)
    checkpoint = torch.load(cnn_model_path, map_location=DEVICE, weights_only=False)
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ CNN model loaded")
    
    with open(pkl_model_path, 'rb') as f:
        sklearn_model = pickle.load(f)
    print(f"  ✓ Sklearn model loaded")
    
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.nii')])
    
    print(f"\n📊 Testing {len(test_files)} files...\n")
    print("="*100)
    
    for i, filename in enumerate(test_files, 1):
        filepath = os.path.join(test_dir, filename)
        parsed = parse_filename(filename)
        if parsed is None:
            continue
        
        true_age, true_sex, true_tissue = parsed
        cnn_age, cnn_sex, cnn_tissue = predict_cnn(cnn_model, filepath)
        sklearn_age, sklearn_sex, sklearn_tissue = predict_sklearn(sklearn_model, filepath)
        
        print(f"[{i:2d}] {filename}")
        print(f"     FROM FILENAME: Age={true_age}  Sex={true_sex}  Tissue={true_tissue}")
        print(f"     CNN PREDICTS:  Age={cnn_age}  Sex={cnn_sex} {'✓' if cnn_sex==true_sex else '✗ (wrong)'}  "
              f"Tissue={cnn_tissue} {'✓' if cnn_tissue==true_tissue else '✗'}")
        print(f"     SKLEARN:       Age={sklearn_age}  Sex={sklearn_sex} {'✓' if sklearn_sex==true_sex else '✗'}  "
              f"Tissue={sklearn_tissue} {'✓' if sklearn_tissue==true_tissue else '✗'}")
        print()
    
    print("="*100)
    print("✅ Demo complete! Use these files in the GUI for testing.")
    print("="*100)


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    demo_predictions(
        os.path.join(SCRIPT_DIR, "TestData"),
        os.path.join(SCRIPT_DIR, "brain_cnn_model.pth"),
        os.path.join(SCRIPT_DIR, "brain_mri_model.pkl")
    )

