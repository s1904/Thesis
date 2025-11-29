"""
Brain MRI Prediction Script
============================
Use the trained model to predict Age, Sex, and Tissue Type
from new brain MRI NIfTI images.

Usage:
    python predict_brain.py <path_to_nii_file>
    python predict_brain.py data/20_F_0010423069_mwp1s0010423069.nii

Or use interactively in Python:
    from predict_brain import predict_image
    result = predict_image("path/to/your/image.nii")
"""

import sys
import pickle
import os

# Import the model class and feature extraction function
from train_brain_model import BrainMRIPredictor, extract_features


def predict_image(nii_path, model_path=None):
    """
    Predict age, sex, and tissue type from a brain MRI NIfTI file.
    
    Parameters:
    -----------
    nii_path : str
        Path to the NIfTI (.nii) file
    model_path : str
        Path to the trained model (.pkl file)
    
    Returns:
    --------
    dict : Dictionary containing predictions
        - predicted_age: float
        - predicted_sex: 'M' or 'F'
        - predicted_tissue_type: 'GM' (gray matter) or 'WM' (white matter)
    """
    # Default model path - check temp directory first
    if model_path is None:
        import tempfile
        temp_model = os.path.join(tempfile.gettempdir(), "brain_mri_model.pkl")
        local_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_mri_model.pkl")

        if os.path.exists(temp_model):
            model_path = temp_model
        elif os.path.exists(local_model):
            model_path = local_model
        else:
            model_path = "brain_mri_model.pkl"

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'!\n"
            f"Please run 'python train_brain_model.py' first to train the model."
        )
    
    # Check if image exists
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Image not found: {nii_path}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make prediction
    print(f"Analyzing image: {nii_path}")
    result = model.predict_from_file(nii_path)
    
    return result


def main():
    """Main function for command-line usage."""
    print("\n" + "="*60)
    print("  BRAIN MRI PREDICTION")
    print("  Predicting: Age, Sex, and Tissue Type")
    print("="*60 + "\n")
    
    # Get image path from command line or ask user
    if len(sys.argv) > 1:
        nii_path = sys.argv[1]
    else:
        print("Usage: python predict_brain.py <path_to_nii_file>")
        print("\nExample files in data folder:")
        
        # Show some example files
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.endswith('.nii') and ('_mwp1' in f or '_mwp2' in f)][:5]
            for f in files:
                print(f"  - data/{f}")
        
        print("\nEnter path to NIfTI file (or 'q' to quit):")
        nii_path = input("> ").strip()
        
        if nii_path.lower() == 'q':
            print("Exiting...")
            return
    
    try:
        # Make prediction
        result = predict_image(nii_path)
        
        # Display results
        print("\n" + "-"*40)
        print("PREDICTION RESULTS:")
        print("-"*40)
        print(f"  Predicted Age:         {result['predicted_age']:.1f} years")
        print(f"  Predicted Sex:         {result['predicted_sex']} ({'Female' if result['predicted_sex'] == 'F' else 'Male'})")
        print(f"  Predicted Tissue Type: {result['predicted_tissue_type']} ({'Gray Matter' if result['predicted_tissue_type'] == 'GM' else 'White Matter'})")
        print("-"*40 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError processing image: {e}")


if __name__ == "__main__":
    main()

