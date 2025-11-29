# Brain MRI Analyzer - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Models](#models)
5. [Grad-CAM Explainability](#grad-cam-explainability)
6. [GUI Application](#gui-application)

---

## 1. Overview

This project predicts three attributes from brain MRI scans:
- **Age** (regression): Predicted age in years
- **Sex** (classification): Male (M) or Female (F)
- **Tissue Type** (classification): Gray Matter (GM) or White Matter (WM)

Two models are available:
1. **3D CNN** - Deep learning model with Grad-CAM visualization
2. **Sklearn** - Traditional ML model (Random Forest + Gradient Boosting)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│              NIfTI MRI File (.nii)                              │
│         Format: {age}_{sex}_{id}_{tissue}.nii                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                                  │
│  1. Load NIfTI with nibabel                                     │
│  2. Resize to target size (64×64×64 for CNN)                    │
│  3. Normalize intensity to [0, 1]                               │
│  4. Convert to PyTorch tensor (CNN) or extract features (ML)    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌──────────┴──────────┐
          ▼                     ▼
┌─────────────────────┐  ┌─────────────────────┐
│     CNN MODEL       │  │   SKLEARN MODEL     │
│  (brain_cnn_model   │  │  (brain_mri_model   │
│      .pth)          │  │      .pkl)          │
│                     │  │                     │
│  3D Convolutions    │  │  Feature Extraction │
│  Multi-task heads   │  │  + Random Forest    │
│  + Grad-CAM         │  │  + Gradient Boost   │
└─────────┬───────────┘  └──────────┬──────────┘
          │                         │
          └──────────┬──────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│   • Predicted Age (years)                                       │
│   • Predicted Sex (M/F)                                         │
│   • Predicted Tissue Type (GM/WM)                               │
│   • Grad-CAM Heatmap (CNN only)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 File Naming Convention

Files must follow this pattern:
```
{age}_{sex}_{subject_id}_{processing_type}{scan_id}.nii
```

| Component | Description | Example |
|-----------|-------------|---------|
| `age` | Subject age in years | `20`, `35`, `65` |
| `sex` | M (Male) or F (Female) | `M`, `F` |
| `subject_id` | Unique subject identifier | `0010423069` |
| `processing_type` | `mwp1` (Gray Matter) or `mwp2` (White Matter) | `mwp1`, `mwp2` |

**Examples:**
- `20_F_0010423069_mwp1s0010423069.nii` → 20yo Female, Gray Matter
- `45_M_0012345678_mwp2s0012345678.nii` → 45yo Male, White Matter

### 3.2 Preprocessing Steps

#### For CNN Model:
```python
# 1. Load NIfTI file
img = nibabel.load(filepath)
data = img.get_fdata().astype(np.float32)

# 2. Resize to 64×64×64 using scipy zoom
factors = [64/s for s in data.shape]
data = scipy.ndimage.zoom(data, factors, order=1)

# 3. Normalize to [0, 1]
data = (data - data.min()) / (data.max() - data.min())

# 4. Add batch and channel dimensions
tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64, 64]
```

#### For Sklearn Model:
```python
# 1. Load NIfTI file
img = nibabel.load(filepath)
data = img.get_fdata()

# 2. Extract brain voxels (non-background)
brain_voxels = data.flatten()[data.flatten() > 0.01]

# 3. Compute 41 statistical features:
features = [
    mean, std, median, min, max,
    percentiles (5, 10, 25, 75, 90, 95),
    sum, count, variance, range, IQR,
    skewness, kurtosis, coefficient_of_variation,
    energy, entropy,
    histogram (20 bins)
]
```

---

## 4. Models

### 4.1 CNN Model Architecture (BrainCNN3D)

```
Input: [1, 1, 64, 64, 64] (batch, channel, depth, height, width)
       │
       ▼
┌──────────────────────────────────────┐
│  Block 1: Conv3d(1→32) + BN + ReLU   │
│           MaxPool3d(2)               │
│  Output: [1, 32, 32, 32, 32]         │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Block 2: Conv3d(32→64) + BN + ReLU  │
│           MaxPool3d(2)               │
│  Output: [1, 64, 16, 16, 16]         │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Block 3: Conv3d(64→128) + BN + ReLU │
│           MaxPool3d(2)               │
│  Output: [1, 128, 8, 8, 8]           │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Block 4: Conv3d(128→256) + BN + ReLU│  ← Grad-CAM Target Layer
│           AdaptiveAvgPool3d(1)       │
│  Output: [1, 256, 1, 1, 1]           │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Shared FC: Linear(256→128) + ReLU   │
│             Dropout(0.5)             │
│  Output: [1, 128]                    │
└──────────────────────────────────────┘
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ Age Head   │ │ Sex Head   │ │Tissue Head │
│ Linear(1)  │ │ Linear(1)  │ │ Linear(1)  │
│            │ │ + Sigmoid  │ │ + Sigmoid  │
└────────────┘ └────────────┘ └────────────┘
```

**Training Configuration:**
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T_0=50)
- Loss Functions:
  - Age: MSE Loss (Mean Squared Error)
  - Sex: BCE Loss (Binary Cross-Entropy)
  - Tissue: BCE Loss
- Batch Size: 4-8
- Epochs: 200

**Data Augmentation (Training):**
- Random horizontal/vertical flips (50% probability)
- Random Gaussian noise (σ=0.02)
- Random intensity scaling (0.9-1.1×)

### 4.2 Sklearn Model Architecture

```
┌─────────────────────────────────────────┐
│         Feature Vector (41 features)    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│           StandardScaler                │
│    (normalize features to μ=0, σ=1)     │
└─────────────────────┬───────────────────┘
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Age       │ │    Sex      │ │   Tissue    │
│  Gradient   │ │   Random    │ │   Random    │
│  Boosting   │ │   Forest    │ │   Forest    │
│  Regressor  │ │ Classifier  │ │ Classifier  │
│             │ │             │ │             │
│ n_est=300   │ │ n_est=200   │ │ n_est=200   │
│ lr=0.05     │ │ max_dep=10  │ │             │
│ max_dep=5   │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

## 5. Grad-CAM Explainability

### 5.1 What is Grad-CAM?

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainability technique that highlights which regions of an image are most important for the model's prediction.

### 5.2 How it Works

```
Step 1: Forward Pass
─────────────────────
Input Image → CNN → Prediction
                ↓
        Store activations (A) from target layer

Step 2: Backward Pass
─────────────────────
Prediction → Backpropagate → Gradients
                                ↓
                    Store gradients (∂y/∂A) from target layer

Step 3: Compute Weights
─────────────────────
weights = GlobalAveragePool(gradients)
        = (1/Z) × Σᵢ Σⱼ Σₖ (∂y/∂Aᵢⱼₖ)

Step 4: Generate Heatmap
─────────────────────
CAM = ReLU(Σ weights × activations)
    = ReLU(Σₖ αₖ × Aₖ)

Step 5: Upsample & Overlay
─────────────────────
Resize CAM to original image size
Apply colormap (blue→green→yellow→red)
Overlay on brain MRI slices
```

### 5.3 Implementation

```python
class GradCAM3D:
    def __init__(self, model, target_layer):
        # Register hooks to capture activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def generate_cam(self, input_tensor, target='age'):
        # Forward pass - get prediction
        age, sex, tissue = self.model(input_tensor)

        # Select which output to explain
        output = age  # or sex, tissue

        # Backward pass - compute gradients
        output.backward()

        # Compute importance weights
        weights = torch.mean(self.gradients, dim=(2,3,4))

        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
```

### 5.4 Visualization

The heatmap is displayed in three anatomical views:

| View | Plane | Description |
|------|-------|-------------|
| **Axial** | X-Y (horizontal) | Top-down view of brain |
| **Coronal** | X-Z (frontal) | Front view of brain |
| **Sagittal** | Y-Z (side) | Side view of brain |

**Color Scale:**
- 🔵 **Blue** = Low importance (0.0 - 0.2)
- 🟢 **Green** = Medium-low importance (0.2 - 0.4)
- 🟡 **Yellow** = Medium-high importance (0.4 - 0.7)
- 🔴 **Red** = High importance (0.7 - 1.0)

---

## 6. GUI Application

### 6.1 Components

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠 Brain MRI Analyzer                                          │
│  Predict Age, Sex & Tissue Type from MRI Scans                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Model Dropdown ▼]  [📁 Load MRI]  [🔍 Predict]  [💾 Save]    │
│  ✅ Grad-CAM Available                                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │             │  │             │  │             │             │
│  │   AXIAL     │  │  CORONAL    │  │  SAGITTAL   │             │
│  │   VIEW      │  │   VIEW      │  │   VIEW      │             │
│  │             │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     PREDICTION RESULTS                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │ 👤 Age        │ │ ⚧ Sex         │ │ 🧬 Tissue     │         │
│  │   32 years    │ │   Female      │ │  Gray Matter  │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
│                                                                 │
│  Loaded: subject_001_mwp1.nii                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Workflow

```
1. Launch Application
   └── python brain_mri_gui.py

2. Select Model
   ├── CNN (.pth) - With Grad-CAM [Recommended]
   └── Sklearn (.pkl) - Faster (No Grad-CAM)

3. Load MRI Image
   └── Click "Load MRI Image" → Select .nii file

4. Run Prediction
   └── Click "Predict & Explain"
       ├── Preprocesses image
       ├── Runs model inference
       ├── Generates Grad-CAM (if CNN)
       └── Displays results

5. View Results
   ├── Age prediction (years)
   ├── Sex prediction (Male/Female)
   ├── Tissue type (Gray/White Matter)
   └── Grad-CAM heatmaps (3 views)

6. Save Results (Optional)
   └── Click "Save Results" → Export to file
```

### 6.3 Brain Mask Detection

To ensure the heatmap only appears inside the brain (not background):

```python
def create_brain_mask(slice_data):
    # 1. Threshold to separate brain from background
    threshold = percentile(nonzero_voxels, 25)
    mask = data > threshold

    # 2. Fill holes inside brain
    mask = binary_fill_holes(mask)

    # 3. Keep only largest connected component
    labeled, num = label(mask)
    sizes = [sum(labeled == i) for i in range(1, num+1)]
    mask = (labeled == argmax(sizes) + 1)

    # 4. Erode to keep heatmap inside boundary
    mask = binary_erosion(mask, iterations=3)

    # 5. Smooth edges
    mask = gaussian_filter(mask, sigma=2)

    return mask
```

---

## 7. File Structure

```
Thesis/
├── brain_mri_gui.py        # Main GUI application
├── train_brain_cnn.py      # CNN training script
├── train_brain_model.py    # Sklearn training script
├── predict_brain_cnn.py    # CNN prediction CLI
├── predict_brain.py        # Sklearn prediction CLI
├── brain_cnn_model.pth     # Trained CNN model weights
├── brain_mri_model.pkl     # Trained Sklearn model
├── Data/                   # Training data directory
│   ├── 20_F_xxx_mwp1xxx.nii
│   ├── 20_F_xxx_mwp2xxx.nii
│   └── ...
├── README.md               # Quick start guide
└── DOCUMENTATION.md        # This file
```

---

## 8. Performance Metrics

| Model | Age MAE | Sex Accuracy | Tissue Accuracy | Inference Time |
|-------|---------|--------------|-----------------|----------------|
| CNN   | ~5 years | ~70% | ~98% | ~2-3 seconds |
| Sklearn | ~8 years | ~64% | ~96% | <1 second |

---

## 9. Dependencies

```
numpy>=1.20.0
nibabel>=3.0.0
scipy>=1.7.0
scikit-learn>=1.0.0
torch>=1.9.0
matplotlib>=3.4.0
pillow>=8.0.0
tqdm>=4.60.0
```

---

## 10. References

- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- NIfTI file format: https://nifti.nimh.nih.gov/
- PyTorch Documentation: https://pytorch.org/docs/

