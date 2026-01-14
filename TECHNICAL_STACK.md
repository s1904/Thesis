# 🧠 Brain MRI Analyzer - Technical Stack Documentation

## 📋 Project Overview

**Project Name:** Brain MRI Analyzer with Explainable AI  
**Purpose:** Multi-task prediction (Age, Sex, Tissue Type) from brain MRI scans with visual explanations  
**Platform:** Windows Desktop Application  

---

## 🛠️ Technology Stack

### 1. Programming Language
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Primary development language |

---

### 2. Deep Learning Framework

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.0+ | Deep learning framework for 3D CNN |
| **torch.nn** | - | Neural network modules (Conv3d, BatchNorm3d, ReLU, etc.) |
| **torch.optim** | - | Optimizers (Adam with weight decay) |
| **torch.nn.functional** | - | Functional operations (relu, sigmoid) |
| **CUDA** | 11.x/12.x | GPU acceleration (optional) |

---

### 3. Machine Learning (Traditional)

| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | 1.3+ | Traditional ML models |
| **RandomForestClassifier** | - | Sex & Tissue classification |
| **GradientBoostingRegressor** | - | Age regression |
| **StandardScaler** | - | Feature normalization |
| **LabelEncoder** | - | Categorical encoding |

---

### 4. Medical Imaging

| Library | Version | Purpose |
|---------|---------|---------|
| **NiBabel** | 5.0+ | NIfTI file reading (.nii, .nii.gz) |
| **SciPy (ndimage)** | 1.11+ | 3D image resampling, interpolation, morphological operations |

---

### 5. GUI Framework

| Library | Version | Purpose |
|---------|---------|---------|
| **Tkinter** | Built-in | Main GUI framework |
| **ttk** | Built-in | Themed widgets (Combobox, Scale, etc.) |
| **filedialog** | Built-in | File selection dialogs |
| **messagebox** | Built-in | Alert/warning dialogs |

---

### 6. Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| **Matplotlib** | 3.7+ | Scientific plotting |
| **FigureCanvasTkAgg** | - | Embedding matplotlib in Tkinter |
| **PIL/Pillow** | 10.0+ | Image processing for display |
| **Custom Colormaps** | - | Grad-CAM heatmap visualization |

---

### 7. Data Processing

| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | 1.24+ | Numerical computations, array operations |
| **SciPy** | 1.11+ | Scientific computing, image processing |
| **tqdm** | 4.65+ | Progress bars for training |

---

### 8. Serialization

| Library | Version | Purpose |
|---------|---------|---------|
| **pickle** | Built-in | Sklearn model serialization (.pkl) |
| **torch.save/load** | - | PyTorch model checkpoints (.pth) |

---

## 🏗️ Architecture Components

### Neural Network Architecture (3D CNN)

```
Input: 3D MRI Volume (64×64×64×1)
    │
    ▼
┌─────────────────────────────────┐
│ Block 1: Conv3d(1→32) + BN + ReLU │
│          Conv3d(32→32) + BN + ReLU│
│          MaxPool3d(2) + Dropout   │
└─────────────────────────────────┘
    │ (32×32×32×32)
    ▼
┌─────────────────────────────────┐
│ Block 2: Conv3d(32→64) + BN + ReLU│
│          Conv3d(64→64) + BN + ReLU│
│          MaxPool3d(2) + Dropout   │
└─────────────────────────────────┘
    │ (16×16×16×64)
    ▼
┌─────────────────────────────────┐
│ Block 3: Conv3d(64→128) + BN + ReLU│
│          Conv3d(128→128) + BN + ReLU│
│          MaxPool3d(2) + Dropout    │
└─────────────────────────────────┘
    │ (8×8×8×128)
    ▼
┌─────────────────────────────────┐
│ Block 4: Conv3d(128→256) + BN + ReLU│
│          Conv3d(256→256) + BN + ReLU│
│          MaxPool3d(2) + Dropout    │
└─────────────────────────────────┘
    │ (4×4×4×256)
    ▼
┌─────────────────────────────────┐
│ Block 5: Conv3d(256→512) + BN + ReLU│
│          AdaptiveAvgPool3d(1)      │
└─────────────────────────────────┘
    │ (1×1×1×512)
    ▼
┌─────────────────────────────────┐
│ Shared FC: Linear(512→256→128)    │
│            + BN + ReLU + Dropout   │
└─────────────────────────────────┘
    │
    ├──────────┬──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Age Head│ │Sex Head│ │Tissue  │
│Lin→64→1│ │Lin→64→1│ │Lin→64→1│
└────────┘ └────────┘ └────────┘
    │          │          │
    ▼          ▼          ▼
  Age      Sex(σ)    Tissue(σ)
(0-1)      (0-1)      (0-1)
```

---

## 🔬 Explainable AI (XAI) Components

### Grad-CAM (Gradient-weighted Class Activation Mapping)

| Component | Description |
|-----------|-------------|
| **Target Layer** | Block 5 Conv3d (512 channels) |
| **Forward Hook** | Captures activations during forward pass |
| **Backward Hook** | Captures gradients during backward pass |
| **CAM Generation** | Weighted sum of activations by gradient importance |
| **Visualization** | 3-view display (Axial, Coronal, Sagittal) |

---

## 📁 Data Format

### Input Data Specifications

| Attribute | Value |
|-----------|-------|
| **File Format** | NIfTI (.nii, .nii.gz) |
| **Dimensions** | 3D volumetric (typically 121×145×121) |
| **Preprocessing** | CAT12 VBM (SPM) |
| **Tissue Types** | mwp1/m0wrp1s (Gray Matter), mwp2/m0wrp2s (White Matter) |

### Filename Convention
```
{Age}_{Sex}_{ID}_{ProcessingType}.nii

Examples:
- 18_F_m0wrp1sB01_1996EF2006-0002-00001-000192-01.nii
- 20_M_0010420366_mwp1s0010420366.nii
```

---

## 📊 Model Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| **Age** | Regression | 0-100 years | Predicted brain age |
| **Sex** | Binary Classification | M/F | Biological sex |
| **Tissue** | Binary Classification | GM/WM | Gray/White matter |

---

## 🖥️ GUI Features

| Feature | Technology | Description |
|---------|------------|-------------|
| **MRI Viewer** | Tkinter Canvas + PIL | Interactive slice viewer with slider |
| **Model Selector** | ttk.Combobox | Switch between CNN and Sklearn models |
| **Grad-CAM Display** | Matplotlib + FigureCanvasTkAgg | 3-view heatmap visualization |
| **Color Legend** | Custom Tkinter Frame | Interactive color interpretation guide |
| **Statistics Panel** | Tkinter Labels | Real-time activation statistics |
| **Tooltips** | Custom ToolTip class | Hover explanations |

---

## 📦 File Structure

```
Thesis/
├── brain_mri_gui.py          # Main GUI application
├── train_brain_cnn.py        # CNN training script
├── train_brain_model.py      # Sklearn training script
├── brain_cnn_model.pth       # Trained CNN weights
├── brain_mri_model.pkl       # Trained Sklearn model
├── demo_predictions.py       # Demo/testing script
├── Data/                     # Training data (NIfTI files)
├── TestData/                 # Test data (20 samples)
├── TECHNICAL_STACK.md        # This document
└── DOCUMENTATION.md          # User documentation
```

---

## ⚙️ System Requirements

### Minimum Requirements
| Component | Specification |
|-----------|---------------|
| **OS** | Windows 10/11 |
| **Python** | 3.11+ |
| **RAM** | 8 GB |
| **Storage** | 2 GB (for models and data) |

### Recommended (for Training)
| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA with CUDA support |
| **VRAM** | 6+ GB |
| **RAM** | 16+ GB |

---

## 🔧 Installation

```bash
# Create environment
conda create -n brain_mri python=3.11
conda activate brain_mri

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy scipy scikit-learn matplotlib pillow tqdm

# Run GUI
python brain_mri_gui.py
```

---

## 📈 Performance Metrics

### CNN Model
| Metric | Value |
|--------|-------|
| Age MAE | ~10.5 years |
| Sex Accuracy | ~55% |
| Tissue Accuracy | ~100% |

### Sklearn Model
| Metric | Value |
|--------|-------|
| Age MAE | ~5-8 years |
| Sex Accuracy | ~95% |
| Tissue Accuracy | ~100% |

---

## 🔗 Key Dependencies Summary

```
torch>=2.0.0
nibabel>=5.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pillow>=10.0.0
tqdm>=4.65.0
```

---

*Document Version: 1.0*
*Last Updated: January 2026*

