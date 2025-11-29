# 🧠 Brain MRI Analyzer

A deep learning project that predicts **Age**, **Sex**, and **Tissue Type** (Gray Matter/White Matter) from brain MRI scans with **Grad-CAM visualization**.

![GUI Screenshot](screenshot.png)

---

## ✨ Features

- **GUI Application** - User-friendly interface to load and analyze MRI scans
- **CNN Model** - 3D Convolutional Neural Network with Grad-CAM explainability
- **Sklearn Model** - Faster traditional ML model for quick predictions
- **Grad-CAM Visualization** - See which brain regions influence predictions
- **Multi-view Display** - Axial, Coronal, and Sagittal brain views

---

## 📦 Requirements

```bash
pip install numpy nibabel scikit-learn torch matplotlib scipy pillow tqdm
```

---

## 🚀 Quick Start - GUI Application

**Run the GUI:**
```bash
python brain_mri_gui.py
```

**How to use:**
1. Click **"Load MRI Image"** to select a `.nii` file
2. Choose model from dropdown:
   - **CNN (.pth)** - Slower but has Grad-CAM visualization
   - **Sklearn (.pkl)** - Faster predictions
3. Click **"Predict & Explain"** to run analysis
4. View results:
   - **Age** - Predicted age in years
   - **Sex** - Male/Female classification
   - **Tissue Type** - Gray Matter (GM) or White Matter (WM)
   - **Grad-CAM** - Heatmap showing important brain regions (CNN only)

---

## 📁 File Naming Convention

Your MRI files should follow this pattern:
```
{age}_{sex}_{id}_{tissue_type}.nii
```

**Examples:**
- `20_F_0010423069_mwp1s0010423069.nii` → 20 years, Female, Gray Matter
- `35_M_0012345678_mwp2s0012345678.nii` → 35 years, Male, White Matter

**Tissue Type Codes:**
- `mwp1` = Gray Matter (GM)
- `mwp2` = White Matter (WM)

---

## 🏋️ Training Models

### Train CNN Model (Recommended)
```bash
python train_brain_cnn.py
```
- Creates `brain_cnn_model.pth`
- Supports Grad-CAM visualization
- Better accuracy

### Train Sklearn Model
```bash
python train_brain_model.py
```
- Creates `brain_mri_model.pkl`
- Faster training and inference
- No Grad-CAM support

---

## 💻 Command Line Prediction

### Using CNN Model
```bash
python predict_brain_cnn.py path/to/image.nii
```

### Using Sklearn Model
```bash
python predict_brain.py path/to/image.nii
```

**Output:**
```
----------------------------------------
PREDICTION RESULTS:
----------------------------------------
  Predicted Age:         20 years
  Predicted Sex:         F (Female)
  Predicted Tissue Type: GM (Gray Matter)
----------------------------------------
```

---

## 🐍 Python API

```python
# Using CNN model
from predict_brain_cnn import predict_image
result = predict_image("path/to/image.nii")

# Using Sklearn model
from predict_brain import predict_image
result = predict_image("path/to/image.nii")

print(f"Age: {result['predicted_age']} years")
print(f"Sex: {result['predicted_sex']}")
print(f"Tissue: {result['predicted_tissue_type']}")
```

---

## 📂 Project Structure

```
Thesis/
├── brain_mri_gui.py       # 🖥️  GUI Application
├── train_brain_cnn.py     # 🏋️  CNN training script
├── train_brain_model.py   # 🏋️  Sklearn training script
├── predict_brain_cnn.py   # 🔮 CNN prediction script
├── predict_brain.py       # 🔮 Sklearn prediction script
├── brain_cnn_model.pth    # 📦 Trained CNN model
├── brain_mri_model.pkl    # 📦 Trained Sklearn model
├── Data/                  # 📁 Training data folder
│   └── *.nii files
└── README.md
```

---

## 📊 Model Performance

| Model | Age MAE | Sex Accuracy | Tissue Accuracy |
|-------|---------|--------------|-----------------|
| CNN   | ~5 years | ~70% | ~98% |
| Sklearn | ~8 years | ~64% | ~96% |

---

## 🎨 Grad-CAM Explanation

The CNN model includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** which highlights brain regions that most influence the model's predictions:

- **Red/Yellow** = High importance regions
- **Blue/Green** = Lower importance regions

This helps understand which anatomical structures the model uses for age, sex, and tissue type prediction.

