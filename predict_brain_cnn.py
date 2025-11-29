"""
Brain MRI 3D-CNN Prediction Script
===================================
Use the trained 3D CNN model to predict Age, Sex, and Tissue Type.

Usage:
    python predict_brain_cnn.py <path_to_nii_file>
    python predict_brain_cnn.py "D:/data/20_F_001217302_mwp2s0011217320.nii"
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom

# Import model class
from train_brain_cnn import BrainCNN3D, DEVICE


def load_model(model_path):
    """Load the trained CNN model."""
    model = BrainCNN3D().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"  - Age MAE: {checkpoint['age_mae']:.2f} years")
    print(f"  - Sex Accuracy: {checkpoint['sex_acc']:.1f}%")
    print(f"  - Tissue Accuracy: {checkpoint['tissue_acc']:.1f}%")
    
    return model


def predict(model, nii_path, target_size=(64, 64, 64)):
    """Predict age, sex, and tissue type from a brain MRI NIfTI file."""
    
    # Load image
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)
    
    # Resize to target size
    factors = [t / s for t, s in zip(target_size, data.shape)]
    data = zoom(data, factors, order=1)
    
    # Normalize to 0-1
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # Convert to tensor [1, 1, D, H, W]
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    # Predict
    with torch.no_grad():
        age_pred, sex_pred, tissue_pred = model(data)
    
    # Convert predictions
    age = int(round(age_pred.item() * 100))
    sex = 'F' if sex_pred.item() > 0.5 else 'M'
    tissue = 'WM' if tissue_pred.item() > 0.5 else 'GM'
    
    return {
        'predicted_age': age,
        'predicted_sex': sex,
        'predicted_tissue_type': tissue,
        'confidence': {
            'sex': abs(sex_pred.item() - 0.5) * 2 * 100,  # 0-100%
            'tissue': abs(tissue_pred.item() - 0.5) * 2 * 100
        }
    }


def main():
    print("\n" + "="*60)
    print("  BRAIN MRI 3D-CNN PREDICTION")
    print("  Deep Learning Model")
    print("="*60 + "\n")
    
    # Get image path
    if len(sys.argv) > 1:
        nii_path = sys.argv[1]
    else:
        print("Usage: python predict_brain_cnn.py <path_to_nii_file>")
        nii_path = input("\nEnter path to NIfTI file: ").strip().strip('"')
    
    # Check if file exists
    if not os.path.exists(nii_path):
        print(f"\nError: File not found: {nii_path}")
        return
    
    # Find model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "brain_cnn_model.pth")
    
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please run 'python train_brain_cnn.py' first to train the model.")
        return
    
    try:
        # Load model
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        
        # Make prediction
        print(f"\nAnalyzing image: {nii_path}")
        result = predict(model, nii_path)
        
        # Display results
        print("\n" + "-"*40)
        print("PREDICTION RESULTS:")
        print("-"*40)
        print(f"  Predicted Age:         {result['predicted_age']} years")
        print(f"  Predicted Sex:         {result['predicted_sex']} ({'Female' if result['predicted_sex'] == 'F' else 'Male'})")
        print(f"  Predicted Tissue Type: {result['predicted_tissue_type']} ({'Gray Matter' if result['predicted_tissue_type'] == 'GM' else 'White Matter'})")
        print("-"*40)
        print(f"  Sex Confidence:        {result['confidence']['sex']:.1f}%")
        print(f"  Tissue Confidence:     {result['confidence']['tissue']:.1f}%")
        print("-"*40 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

