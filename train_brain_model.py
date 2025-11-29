"""
Brain MRI Age, Sex, and Tissue Type Prediction Model
====================================================
This script trains a machine learning model to predict:
- Age (regression)
- Sex (classification: M/F)
- Tissue Type (classification: GM=gray matter / WM=white matter)

From preprocessed brain MRI NIfTI files (mwp1=GM, mwp2=WM)
"""

import os
import re
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def parse_filename(filename):
    """
    Parse filename to extract age, sex, and tissue type.
    Example: 20_F_001217302_mwp1s0011217320.nii
    Returns: (age, sex, tissue_type) or None if not matching pattern
    """
    # Match pattern: Age_Sex_ID_ProcessingType
    # We only want mwp1 (gray matter) and mwp2 (white matter) files
    pattern = r'^(\d+)_([MF])_\d+_(mwp[12])'
    match = re.match(pattern, filename)
    
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue_prefix = match.group(3)
        tissue_type = 'GM' if tissue_prefix == 'mwp1' else 'WM'
        return age, sex, tissue_type
    return None


def extract_features(nii_path):
    """
    Extract statistical features from a NIfTI brain image.
    Returns a feature vector for the image.
    """
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Flatten and remove zero/background voxels
        flat_data = data.flatten()
        brain_voxels = flat_data[flat_data > 0.01]  # threshold to get brain tissue
        
        if len(brain_voxels) == 0:
            brain_voxels = flat_data[flat_data > 0]
        
        if len(brain_voxels) == 0:
            return None
        
        # Extract statistical features
        features = [
            np.mean(brain_voxels),           # Mean intensity
            np.std(brain_voxels),            # Standard deviation
            np.median(brain_voxels),         # Median
            np.min(brain_voxels),            # Min
            np.max(brain_voxels),            # Max
            np.percentile(brain_voxels, 5),  # 5th percentile
            np.percentile(brain_voxels, 10), # 10th percentile
            np.percentile(brain_voxels, 25), # 25th percentile
            np.percentile(brain_voxels, 75), # 75th percentile
            np.percentile(brain_voxels, 90), # 90th percentile
            np.percentile(brain_voxels, 95), # 95th percentile
            np.sum(brain_voxels),            # Total intensity (proxy for volume)
            len(brain_voxels),               # Number of brain voxels
            np.var(brain_voxels),            # Variance
            np.max(brain_voxels) - np.min(brain_voxels),  # Range
            np.percentile(brain_voxels, 75) - np.percentile(brain_voxels, 25),  # IQR
            # Skewness and Kurtosis
            ((brain_voxels - np.mean(brain_voxels))**3).mean() / (np.std(brain_voxels)**3 + 1e-10),
            ((brain_voxels - np.mean(brain_voxels))**4).mean() / (np.std(brain_voxels)**4 + 1e-10),
            # Coefficient of variation
            np.std(brain_voxels) / (np.mean(brain_voxels) + 1e-10),
            # Energy and entropy-like features
            np.sum(brain_voxels**2),         # Energy
            -np.sum((brain_voxels/brain_voxels.sum()) * np.log(brain_voxels/brain_voxels.sum() + 1e-10)),  # Entropy
        ]

        # Add histogram features (20 bins for more detail)
        hist, _ = np.histogram(brain_voxels, bins=20, density=True)
        features.extend(hist.tolist())
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error processing {nii_path}: {e}")
        return None


def load_dataset(data_dir):
    """
    Load all mwp1 (GM) and mwp2 (WM) files from data directory.
    Returns features array and labels.
    """
    print(f"Loading data from: {data_dir}")
    
    features_list = []
    ages = []
    sexes = []
    tissue_types = []
    filenames = []
    
    # Get all .nii files
    nii_files = [f for f in os.listdir(data_dir) if f.endswith('.nii')]
    
    # Filter to only mwp1 and mwp2 files
    mwp_files = [f for f in nii_files if '_mwp1' in f or '_mwp2' in f]
    
    print(f"Found {len(mwp_files)} GM/WM files to process...")
    
    for filename in tqdm(mwp_files, desc="Extracting features"):
        parsed = parse_filename(filename)
        if parsed is None:
            continue
        
        age, sex, tissue_type = parsed
        filepath = os.path.join(data_dir, filename)
        
        features = extract_features(filepath)
        if features is not None:
            features_list.append(features)
            ages.append(age)
            sexes.append(sex)
            tissue_types.append(tissue_type)
            filenames.append(filename)
    
    print(f"\nSuccessfully loaded {len(features_list)} samples")

    return (np.array(features_list),
            np.array(ages),
            np.array(sexes),
            np.array(tissue_types),
            filenames)


class BrainMRIPredictor:
    """
    Multi-output predictor for brain MRI images.
    Predicts: Age, Sex, and Tissue Type
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.sex_encoder = LabelEncoder()
        self.tissue_encoder = LabelEncoder()

        # Models for each prediction task
        # Use GradientBoosting for age (better for regression) with tuned parameters
        self.age_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        self.sex_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
        self.tissue_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

        self.is_trained = False

    def fit(self, X, ages, sexes, tissue_types):
        """Train all models."""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        sex_encoded = self.sex_encoder.fit_transform(sexes)
        tissue_encoded = self.tissue_encoder.fit_transform(tissue_types)

        # Split data
        X_train, X_test, age_train, age_test, sex_train, sex_test, tissue_train, tissue_test = \
            train_test_split(X_scaled, ages, sex_encoded, tissue_encoded,
                           test_size=0.2, random_state=42)

        # Train Age model (regression)
        print("\n[1/3] Training Age Prediction Model...")
        self.age_model.fit(X_train, age_train)
        age_pred = self.age_model.predict(X_test)
        age_mae = mean_absolute_error(age_test, age_pred)
        print(f"      Age Model MAE: {age_mae:.2f} years")

        # Train Sex model (classification)
        print("\n[2/3] Training Sex Prediction Model...")
        self.sex_model.fit(X_train, sex_train)
        sex_pred = self.sex_model.predict(X_test)
        sex_acc = accuracy_score(sex_test, sex_pred)
        print(f"      Sex Model Accuracy: {sex_acc*100:.2f}%")

        # Train Tissue model (classification)
        print("\n[3/3] Training Tissue Type Prediction Model...")
        self.tissue_model.fit(X_train, tissue_train)
        tissue_pred = self.tissue_model.predict(X_test)
        tissue_acc = accuracy_score(tissue_test, tissue_pred)
        print(f"      Tissue Type Model Accuracy: {tissue_acc*100:.2f}%")

        self.is_trained = True

        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)

        return {
            'age_mae': age_mae,
            'sex_accuracy': sex_acc,
            'tissue_accuracy': tissue_acc
        }

    def predict(self, X):
        """Predict age, sex, and tissue type for given features."""
        if not self.is_trained:
            raise ValueError("Model not trained! Call fit() first.")

        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)

        age_pred = self.age_model.predict(X_scaled)
        sex_pred = self.sex_encoder.inverse_transform(self.sex_model.predict(X_scaled))
        tissue_pred = self.tissue_encoder.inverse_transform(self.tissue_model.predict(X_scaled))

        return age_pred, sex_pred, tissue_pred

    def predict_from_file(self, nii_path):
        """Predict from a NIfTI file path."""
        features = extract_features(nii_path)
        if features is None:
            raise ValueError(f"Could not extract features from {nii_path}")

        age, sex, tissue = self.predict(features)
        return {
            'predicted_age': int(round(age[0])),
            'predicted_sex': sex[0],
            'predicted_tissue_type': tissue[0]
        }


def main():
    """Main training function."""
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "brain_mri_model.pkl")

    print("\n" + "="*60)
    print("  BRAIN MRI PREDICTION MODEL TRAINING")
    print("  Predicting: Age, Sex, and Tissue Type (GM/WM)")
    print("="*60)

    # Load dataset
    X, ages, sexes, tissue_types, filenames = load_dataset(DATA_DIR)

    # Print dataset statistics
    print("\n" + "-"*40)
    print("DATASET STATISTICS:")
    print("-"*40)
    print(f"Total samples: {len(X)}")
    print(f"Age range: {ages.min()} - {ages.max()} years")
    print(f"Sex distribution: M={sum(sexes=='M')}, F={sum(sexes=='F')}")
    print(f"Tissue distribution: GM={sum(tissue_types=='GM')}, WM={sum(tissue_types=='WM')}")
    print(f"Feature vector size: {X.shape[1]}")

    # Create and train model
    model = BrainMRIPredictor()
    metrics = model.fit(X, ages, sexes, tissue_types)

    # Save model
    print(f"\nSaving model to: {MODEL_PATH}")

    # Serialize model to bytes first
    model_bytes = pickle.dumps(model)

    # Write bytes to file
    with open(MODEL_PATH, 'wb') as f:
        f.write(model_bytes)

    print(f"Model saved successfully!")
    print(f"File size: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")

    # Test prediction on a sample
    print("\n" + "-"*40)
    print("SAMPLE PREDICTION TEST:")
    print("-"*40)
    sample_idx = 0
    sample_features = X[sample_idx]
    age_pred, sex_pred, tissue_pred = model.predict(sample_features)

    print(f"File: {filenames[sample_idx]}")
    print(f"Actual    -> Age: {ages[sample_idx]}, Sex: {sexes[sample_idx]}, Tissue: {tissue_types[sample_idx]}")
    print(f"Predicted -> Age: {int(round(age_pred[0]))}, Sex: {sex_pred[0]}, Tissue: {tissue_pred[0]}")

    print("\n" + "="*60)
    print(f"  MODEL READY! Use '{MODEL_PATH}' for predictions")
    print("="*60)


if __name__ == "__main__":
    main()

