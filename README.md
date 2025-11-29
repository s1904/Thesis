# Brain MRI Analysis - Age, Sex & Tissue Type Prediction

A machine learning project that predicts **Age**, **Sex**, and **Tissue Type** (Gray Matter/White Matter) from brain MRI scans.

## Requirements

```bash
pip install numpy nibabel scikit-learn joblib tqdm
```

## File Naming Convention

Your MRI files must follow this naming pattern:
```
{age}_{sex}_{id}_{tissue_type}.nii
```

**Examples:**
- `20_F_0010423069_mwp1s0010423069.nii` → 20 years old, Female, Gray Matter
- `35_M_0012345678_mwp2s0012345678.nii` → 35 years old, Male, White Matter

**Tissue Type Codes:**
- `mwp1` = Gray Matter (GM)
- `mwp2` = White Matter (WM)

---

## 1. Training a New Model

Place your training data (`.nii` files) in the `data/` folder, then run:

```bash
python train_brain_model.py
```

**Output:**
- Creates `brain_mri_model.pkl` (trained model file)
- Shows training statistics and accuracy metrics

**Custom data directory:**
```python
# In Python
from train_brain_model import BrainMRIPredictor, load_dataset

# Load data from custom directory
X, y_age, y_sex, y_tissue, filenames = load_dataset("D:/your/data/folder")

# Train model
model = BrainMRIPredictor()
model.train(X, y_age, y_sex, y_tissue)

# Save model
model.save("my_custom_model.pkl")
```

---

## 2. Making Predictions

### Command Line

```bash
python predict_brain.py path/to/your/image.nii
```

**Example:**
```bash
python predict_brain.py "D:\W.M-G.M\clean_dataset_all\20_F_001217302_mwp2s0011217320.nii"
```

**Output:**
```
----------------------------------------
PREDICTION RESULTS:
----------------------------------------
  Predicted Age:         20.3 years
  Predicted Sex:         F (Female)
  Predicted Tissue Type: WM (White Matter)
----------------------------------------
```

### Python Script

```python
from predict_brain import predict_image

# Single prediction
result = predict_image("path/to/image.nii")

print(f"Age: {result['predicted_age']:.1f} years")
print(f"Sex: {result['predicted_sex']}")
print(f"Tissue: {result['predicted_tissue_type']}")
```

### Using the .pkl Model Directly

```python
import pickle

# Load the trained model
with open("brain_mri_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict from a file
result = model.predict_from_file("path/to/image.nii")

print(result)
# {'predicted_age': 20.3, 'predicted_sex': 'F', 'predicted_tissue_type': 'WM'}
```

---

## 3. Batch Predictions

```python
import os
from predict_brain import predict_image

folder = "D:/my_brain_scans/"
for filename in os.listdir(folder):
    if filename.endswith(".nii"):
        filepath = os.path.join(folder, filename)
        result = predict_image(filepath)
        print(f"{filename}: Age={result['predicted_age']:.1f}, Sex={result['predicted_sex']}, Tissue={result['predicted_tissue_type']}")
```

---

## Project Structure

```
Thesis/
├── train_brain_model.py   # Training script
├── predict_brain.py       # Prediction script
├── brain_mri_model.pkl    # Trained model (after training)
├── data/                  # Training data folder
│   └── *.nii files
└── README.md
```

---

## Model Performance

| Task | Metric | Score |
|------|--------|-------|
| Age Prediction | MAE | ~8 years |
| Sex Classification | Accuracy | ~64% |
| Tissue Type (GM/WM) | Accuracy | ~96% |

