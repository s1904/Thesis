"""Check the data labels to verify encoding."""
import os
import pandas as pd

# Load the CSV
df = pd.read_csv('mri_dataset.csv')

print("="*60)
print("DATA LABEL VERIFICATION")
print("="*60)

print(f"\nTotal samples: {len(df)}")

print("\n" + "="*60)
print("GENDER DISTRIBUTION")
print("="*60)
print(df['gender'].value_counts())
print("\nPercentages:")
print(df['gender'].value_counts(normalize=True) * 100)

print("\n" + "="*60)
print("TISSUE DISTRIBUTION")
print("="*60)
print(df['tissue'].value_counts())
print("\nPercentages:")
print(df['tissue'].value_counts(normalize=True) * 100)

print("\n" + "="*60)
print("SAMPLE DATA (first 10 rows)")
print("="*60)
print(df[['subject_id', 'tissue', 'gender', 'age']].head(10))

print("\n" + "="*60)
print("ENCODING CHECK")
print("="*60)
print("Expected encoding:")
print("  Gender: Female=0, Male=1")
print("  Tissue: GM=0, WM=1")

# Check actual encoding in the dataset
data_dir = 'brain_mri_data'
sample_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            sample_files.append(os.path.join(root, file))
            if len(sample_files) >= 5:
                break
    if len(sample_files) >= 5:
        break

print(f"\nFound {len(sample_files)} sample files")
print("\nSample filenames:")
for f in sample_files[:5]:
    basename = os.path.basename(f)
    print(f"  {basename}")
    
    # Try to extract info from filename
    if 'Female' in basename or 'female' in basename:
        print(f"    -> Contains 'Female'")
    if 'Male' in basename or 'male' in basename:
        print(f"    -> Contains 'Male'")
    if 'GM' in basename:
        print(f"    -> Contains 'GM'")
    if 'WM' in basename:
        print(f"    -> Contains 'WM'")

