"""Debug script to check if labels are being loaded correctly."""
import os
import numpy as np
import torch
from train_brain_cnn import BrainMRIDataset

# Create dataset
dataset = BrainMRIDataset('brain_mri_data', augment=False)

print("="*60)
print("LABEL DEBUGGING")
print("="*60)

print(f"\nTotal samples in dataset: {len(dataset)}")

# Check first 20 samples
print("\nFirst 20 samples:")
print(f"{'Index':<6} {'Filename':<50} {'Age':<6} {'Sex':<6} {'Tissue':<6}")
print("-"*80)

sex_counts = {0: 0, 1: 0}
tissue_counts = {0: 0, 1: 0}

for i in range(min(20, len(dataset))):
    try:
        data, labels = dataset[i]
        age_norm, sex, tissue = labels[0].item(), labels[1].item(), labels[2].item()
        age = age_norm * 100  # Denormalize
        
        # Get filename
        filename = os.path.basename(dataset.file_paths[i])
        
        sex_label = "Female" if sex == 0 else "Male"
        tissue_label = "GM" if tissue == 0 else "WM"
        
        print(f"{i:<6} {filename:<50} {age:<6.0f} {sex_label:<6} {tissue_label:<6}")
        
        sex_counts[int(sex)] += 1
        tissue_counts[int(tissue)] += 1
        
    except Exception as e:
        print(f"{i:<6} ERROR: {e}")

print("\n" + "="*60)
print("LABEL DISTRIBUTION (first 20 samples)")
print("="*60)
print(f"Sex:")
print(f"  Female (0): {sex_counts[0]}")
print(f"  Male (1): {sex_counts[1]}")
print(f"\nTissue:")
print(f"  GM (0): {tissue_counts[0]}")
print(f"  WM (1): {tissue_counts[1]}")

# Check all samples
print("\n" + "="*60)
print("FULL DATASET DISTRIBUTION")
print("="*60)

all_sex_counts = {0: 0, 1: 0}
all_tissue_counts = {0: 0, 1: 0}

for i in range(len(dataset)):
    try:
        _, labels = dataset[i]
        sex, tissue = labels[1].item(), labels[2].item()
        all_sex_counts[int(sex)] += 1
        all_tissue_counts[int(tissue)] += 1
    except:
        pass

print(f"Sex:")
print(f"  Female (0): {all_sex_counts[0]} ({100*all_sex_counts[0]/len(dataset):.1f}%)")
print(f"  Male (1): {all_sex_counts[1]} ({100*all_sex_counts[1]/len(dataset):.1f}%)")
print(f"\nTissue:")
print(f"  GM (0): {all_tissue_counts[0]} ({100*all_tissue_counts[0]/len(dataset):.1f}%)")
print(f"  WM (1): {all_tissue_counts[1]} ({100*all_tissue_counts[1]/len(dataset):.1f}%)")

# Check if labels match filenames
print("\n" + "="*60)
print("FILENAME vs LABEL VERIFICATION")
print("="*60)

mismatches = 0
for i in range(min(50, len(dataset))):
    try:
        _, labels = dataset[i]
        sex = labels[1].item()
        filename = os.path.basename(dataset.file_paths[i]).lower()
        
        # Check if filename contains gender info
        has_female = 'female' in filename or '_f_' in filename
        has_male = 'male' in filename or '_m_' in filename
        
        if has_female and sex == 1:
            print(f"MISMATCH: {filename} -> label says Male but filename says Female")
            mismatches += 1
        elif has_male and sex == 0:
            print(f"MISMATCH: {filename} -> label says Female but filename says Male")
            mismatches += 1
    except:
        pass

if mismatches == 0:
    print("✓ No mismatches found in first 50 samples")
else:
    print(f"✗ Found {mismatches} mismatches!")

