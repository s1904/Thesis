"""
Create Test Set for Brain MRI Demo
===================================
Randomly selects 20 files (10 patient pairs) from the Data folder
and copies them to a TestData folder for demonstration.
"""

import os
import shutil
import random
from collections import defaultdict

def create_test_set(data_dir, test_dir, num_pairs=10):
    """
    Create a test set with num_pairs patient pairs (each pair = GM + WM files).
    Total files = num_pairs * 2
    """
    print("="*60)
    print("  CREATING TEST SET FOR DEMO")
    print("="*60)
    
    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"\n✓ Created directory: {test_dir}")
    else:
        print(f"\n✓ Using existing directory: {test_dir}")
    
    # Get all .nii files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.nii')]
    print(f"\n📊 Total files in Data folder: {len(all_files)}")
    
    # Group files by patient (same patient has multiple tissue types)
    # Extract patient identifier from filename
    patient_groups = defaultdict(list)
    
    for filename in all_files:
        # Extract patient ID (everything before the tissue type marker)
        # Pattern: Age_Sex_ID_TissueType
        if '_mwp1' in filename or '_mwp2' in filename:
            # For mwp files: 20_F_001217302_mwp1...
            patient_id = filename.split('_mwp')[0]
        elif '_m0wrp1s' in filename or '_m0wrp2s' in filename:
            # For m0wrp files: 17_M_m0wrp1sB01...
            parts = filename.split('_')
            if len(parts) >= 2:
                patient_id = f"{parts[0]}_{parts[1]}"  # Age_Sex
            else:
                continue
        else:
            continue
        
        patient_groups[patient_id].append(filename)
    
    print(f"📊 Unique patients found: {len(patient_groups)}")
    
    # Filter to only include patients with both GM and WM files (pairs)
    complete_pairs = {pid: files for pid, files in patient_groups.items() if len(files) >= 2}
    print(f"📊 Patients with complete pairs (GM+WM): {len(complete_pairs)}")
    
    # Randomly select num_pairs patients
    selected_patients = random.sample(list(complete_pairs.keys()), min(num_pairs, len(complete_pairs)))
    
    # Copy files to test directory
    copied_files = []
    print(f"\n📁 Copying {num_pairs} patient pairs to test set...")
    print("-"*60)
    
    for i, patient_id in enumerate(selected_patients, 1):
        files = complete_pairs[patient_id]
        print(f"\n[{i}/{num_pairs}] Patient: {patient_id}")
        
        for filename in files[:2]:  # Take first 2 files (GM and WM)
            src = os.path.join(data_dir, filename)
            dst = os.path.join(test_dir, filename)
            
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  ✓ Copied: {filename}")
            else:
                print(f"  ⊙ Already exists: {filename}")
            
            copied_files.append(filename)
    
    print("\n" + "="*60)
    print("  TEST SET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Total files in test set: {len(copied_files)}")
    print(f"📁 Location: {os.path.abspath(test_dir)}")
    
    # Show sample files
    print(f"\n📋 Sample files in test set:")
    print("-"*60)
    for filename in sorted(copied_files)[:5]:
        print(f"  • {filename}")
    if len(copied_files) > 5:
        print(f"  ... and {len(copied_files) - 5} more files")
    
    return copied_files


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
    TEST_DIR = os.path.join(SCRIPT_DIR, "TestData")
    
    # Create test set with 10 patient pairs (20 files total)
    files = create_test_set(DATA_DIR, TEST_DIR, num_pairs=10)
    
    print("\n" + "="*60)
    print("  READY FOR DEMO!")
    print("="*60)
    print(f"\nYou can now use the files in '{TEST_DIR}' for testing.")
    print("Load them in the GUI to see predictions!")

