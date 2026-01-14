"""
Brain MRI Prediction GUI with Grad-CAM Explainable AI
=====================================================
Interactive GUI to predict Age, Sex, and Tissue Type from brain MRI scans.
Uses 3D CNN model with Grad-CAM to highlight important brain regions.

Run: python brain_mri_gui.py
"""

import os
import re
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import zoom

# Check for required packages
try:
    import nibabel as nib
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pickle
    # Import BrainMRIPredictor for pickle to work
    from train_brain_model import BrainMRIPredictor
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install with: pip install nibabel pillow torch")
    exit(1)


def parse_filename(filename):
    """
    Parse filename to extract ground truth age, sex, and tissue type.
    Works with both mwp1/mwp2 and m0wrp1s/m0wrp2s files.
    """
    basename = os.path.basename(filename)

    # Pattern 1: mwp1 or mwp2 files (e.g., 20_F_001217302_mwp1s...)
    pattern1 = r'^(\d+)_([MF])_\d+_(mwp[12])'
    match = re.match(pattern1, basename)
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue = 'GM' if match.group(3) == 'mwp1' else 'WM'
        return age, sex, tissue

    # Pattern 2: m0wrp1s or m0wrp2s files (e.g., 18_F_m0wrp1sB01_...)
    pattern2 = r'^(\d+)_([MF])_(m0wrp[12]s)'
    match = re.match(pattern2, basename)
    if match:
        age = int(match.group(1))
        sex = match.group(2)
        tissue = 'GM' if 'm0wrp1s' in match.group(3) else 'WM'
        return age, sex, tissue

    return None, None, None

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ToolTip:
    """Create a tooltip for a given widget with hover definition."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.id = None

        widget.bind("<Enter>", self.schedule_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)
        widget.bind("<ButtonPress>", self.hide_tooltip)

    def schedule_tooltip(self, event=None):
        self.cancel_tooltip()
        self.id = self.widget.after(self.delay, self.show_tooltip)

    def cancel_tooltip(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        # Create tooltip frame with styling
        frame = tk.Frame(tw, bg='#2d2d44', bd=1, relief='solid')
        frame.pack()

        label = tk.Label(frame, text=self.text, justify=tk.LEFT,
                        bg='#2d2d44', fg='#ffffff', font=('Segoe UI', 9),
                        padx=10, pady=8, wraplength=300)
        label.pack()

    def hide_tooltip(self, event=None):
        self.cancel_tooltip()
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# Tooltip definitions for medical/statistical terms
TOOLTIP_DEFINITIONS = {
    # Statistics
    'max_activation': "Maximum Activation:\nThe highest intensity value in the Grad-CAM heatmap.\nHigher values (closer to 1.0) indicate stronger model focus.",
    'mean_activation': "Mean Activation:\nAverage activation across all brain voxels.\nIndicates overall model attention distribution.",
    'active_region': "Active Region %:\nPercentage of brain volume with activation > 50%.\nHigher % = broader focus, Lower % = more specific focus.",
    'focus_area': "Focus Area:\nSpatial location of peak activation.\n• Superior/Inferior: Top/Bottom\n• Anterior/Posterior: Front/Back",

    # Anatomical
    'anatomical_region': "Anatomical Region:\nSpecific brain structure at peak activation.\nMapped using MNI atlas coordinates.",
    'expected_tissue': "Expected Tissue:\nTissue type normally found in the focus region.\n• Gray Matter: Neuronal cell bodies\n• White Matter: Myelinated axons",
    'intensity_profile': "Intensity Profile:\nT1 MRI intensity characteristics:\n• Low (CSF): Cerebrospinal fluid\n• Medium (GM): Gray matter\n• High (WM): White matter",
    'gm_wm_ratio': "GM/WM Ratio:\nRatio of gray to white matter voxels in activated region.\n• ~1.0: Equal GM and WM\n• 2-5: Moderately more GM\n• >10: Strongly GM-dominant\n• <0.5: More white matter",

    # Results
    'age': "Predicted Age:\nBiological age estimated from brain morphology.\nBased on cortical thickness, ventricle size, and sulcal patterns.",
    'sex': "Predicted Sex:\nBiological sex inferred from brain structure.\nBased on skull shape, brain volume, and gray/white matter ratios.",
    'tissue': "Tissue Type:\nPrimary tissue the model focuses on.\n• GM: Gray Matter (cortex, nuclei)\n• WM: White Matter (tracts, corpus callosum)",

    # Buttons/Controls
    'load_mri': "Load MRI:\nOpen a NIfTI (.nii or .nii.gz) brain scan file.\nSupports T1-weighted structural MRI.",
    'predict': "Predict:\nRun the model on the loaded MRI scan.\nGenerates Age, Sex, Tissue predictions + Grad-CAM.",
    'gradcam': "Grad-CAM:\nGradient-weighted Class Activation Mapping.\nHighlights brain regions influencing the prediction.",
    'slice_slider': "Slice Navigator:\nScroll through 2D slices of the 3D brain volume.\nDrag to view different depths.",
    'model_selector': "Model Selector:\n• CNN: Deep learning with Grad-CAM visualization\n• Sklearn: Traditional ML (faster, no visualization)",
}


class BrainCNN3D(nn.Module):
    """Improved 3D CNN for multi-task brain MRI prediction with better accuracy."""

    def __init__(self):
        super(BrainCNN3D, self).__init__()

        # Feature extractor (Deeper 3D CNN with residual connections)
        # NOTE: Using inplace=False for ReLU to avoid Grad-CAM errors
        self.features = nn.Sequential(
            # Block 1 - Initial feature extraction
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),  # Changed to False for Grad-CAM compatibility
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),  # 64 -> 32
            nn.Dropout3d(0.1),

            # Block 2 - Deeper features
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),  # 32 -> 16
            nn.Dropout3d(0.1),

            # Block 3 - High-level features
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=False),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),  # 16 -> 8
            nn.Dropout3d(0.2),

            # Block 4 - Abstract features
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=False),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),  # 8 -> 4
            nn.Dropout3d(0.2),

            # Block 5 - Final feature extraction
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
        )

        # Shared fully connected layers with more capacity
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
        )

        # Task-specific heads with deeper networks
        self.age_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.sex_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.tissue_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        age = self.age_head(x)
        sex_logits = self.sex_head(x)  # Return logits
        tissue_logits = self.tissue_head(x)  # Return logits

        # Apply sigmoid for inference (GUI needs probabilities)
        sex = torch.sigmoid(sex_logits)
        tissue = torch.sigmoid(tissue_logits)

        return age, sex, tissue


def disable_inplace_relu(model):
    """Recursively disable inplace operations in ReLU layers for Grad-CAM compatibility."""
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


class GradCAM3D:
    """Grad-CAM for 3D CNN models with improved compatibility."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Clone to avoid inplace modification issues
        self.activations = output.clone().detach()

    def save_gradient(self, module, grad_input, grad_output):
        # Clone to avoid inplace modification issues
        self.gradients = grad_output[0].clone().detach()

    def generate_cam(self, input_tensor, target='age'):
        """Generate Grad-CAM heatmap with error handling."""
        try:
            self.model.eval()

            # Clone input to avoid modification issues
            input_clone = input_tensor.clone().detach().requires_grad_(True)

            # Forward pass
            age, sex, tissue = self.model(input_clone)

            # Select target output
            if target == 'age':
                output = age.clone()
            elif target == 'sex':
                output = sex.clone()
            else:
                output = tissue.clone()

            # Backward pass
            self.model.zero_grad()
            output.backward(retain_graph=True)

            # Check if gradients were captured
            if self.gradients is None or self.activations is None:
                print("Warning: Gradients or activations not captured")
                return np.zeros((8, 8, 8))  # Return small empty heatmap

            # Get weights (global average pooling of gradients)
            weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)

            # Weighted combination of activations
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)  # Only positive contributions

            # Normalize
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            return cam.squeeze().cpu().numpy()

        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            import traceback
            traceback.print_exc()
            # Return a default heatmap on error
            return np.zeros((8, 8, 8))


class BrainMRIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain MRI Analyzer - Grad-CAM")
        self.root.geometry("900x700")  # Smaller, more compact window
        self.root.minsize(800, 600)
        self.root.configure(bg='#1a1a2e')

        # Load model
        self.model = None
        self.sklearn_model = None
        self.grad_cam = None
        self.current_model_type = None
        self.current_image_data = None
        self.current_tensor = None
        self.cam_heatmap = None
        self.resized_data = None
        self.display_data = None  # Higher resolution for display
        self.current_filepath = None
        self.target_size = (64, 64, 64)
        self.display_size = (128, 128, 128)  # Reduced resolution to prevent crashes
        self.load_model()

        # Create UI
        self.create_widgets()

    def load_model(self):
        """Load the selected model (CNN or sklearn)."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cnn_model_path = os.path.join(script_dir, "brain_cnn_model.pth")
        self.pkl_model_path = os.path.join(script_dir, "brain_mri_model.pkl")

        # Check which models are available
        self.cnn_available = os.path.exists(self.cnn_model_path)
        self.pkl_available = os.path.exists(self.pkl_model_path)

        # Default to CNN if available (has Grad-CAM), otherwise pkl
        if self.cnn_available:
            self.load_cnn_model()
        elif self.pkl_available:
            self.load_pkl_model()
        else:
            messagebox.showwarning("Warning", "No model found!\n\nRun train_brain_cnn.py or train_brain_model.py first.")

    def load_cnn_model(self):
        """Load PyTorch CNN model with Grad-CAM support."""
        self.model = BrainCNN3D().to(DEVICE)
        checkpoint = torch.load(self.cnn_model_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # IMPORTANT: Disable inplace ReLU for Grad-CAM compatibility
        disable_inplace_relu(self.model)
        self.model.eval()

        # Setup Grad-CAM on the last conv layer (Block 5 - final conv before pooling)
        # Block 5 is at index 28 (after Block 4's Dropout3d at index 27)
        target_layer = self.model.features[28]  # Block 5 Conv3d (512 channels)
        self.grad_cam = GradCAM3D(self.model, target_layer)
        self.current_model_type = "cnn"

        print(f"CNN Model loaded successfully!")
        print(f"  Age MAE: {checkpoint.get('age_mae', 'N/A'):.2f} years")
        print(f"  Inplace ReLU disabled for Grad-CAM compatibility")

    def load_pkl_model(self):
        """Load sklearn pickle model (no Grad-CAM)."""
        with open(self.pkl_model_path, 'rb') as f:
            self.sklearn_model = pickle.load(f)
        self.model = None
        self.grad_cam = None
        self.current_model_type = "sklearn"
        print(f"Sklearn Model loaded successfully!")

    def switch_model(self, *args):
        """Switch between CNN and sklearn model."""
        selected = self.model_var.get()
        print(f"[DEBUG] Dropdown selected: '{selected}'")
        print(f"[DEBUG] CNN available: {self.cnn_available}, PKL available: {self.pkl_available}")

        if selected == "CNN (.pth) - With Grad-CAM" and self.cnn_available:
            self.load_cnn_model()
            self.gradcam_status.configure(text="✅ Grad-CAM Available", fg='#4CAF50')
            print("[DEBUG] Loaded CNN model")
        elif selected == "Sklearn (.pkl) - Faster" and self.pkl_available:
            self.load_pkl_model()
            self.gradcam_status.configure(text="❌ Grad-CAM Not Available", fg='#FF6B6B')
            print("[DEBUG] Loaded Sklearn model")
        else:
            print(f"[DEBUG] No model loaded - condition not met")

        # Clear previous results (only if widgets exist)
        self.cam_heatmap = None
        if hasattr(self, 'ax1'):
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.clear()
                ax.axis('off')
            self.canvas.draw()

        print(f"[DEBUG] Current model type: {self.current_model_type}")

    def create_widgets(self):
        """Create all GUI widgets with vertical scrollable layout."""
        # Create main canvas with scrollbar
        self.main_canvas = tk.Canvas(self.root, bg='#1a1a2e', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg='#1a1a2e')

        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )

        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Make scrollable frame expand to canvas width
        self.main_canvas.bind('<Configure>', self._on_canvas_configure)

        # Bind mouse wheel for scrolling
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.main_canvas.bind_all("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.main_canvas.bind_all("<Button-5>", self._on_mousewheel)    # Linux scroll down

        # Title
        title_frame = tk.Frame(self.scrollable_frame, bg='#1a1a2e')
        title_frame.pack(pady=(20, 10), fill=tk.X)

        title = tk.Label(title_frame, text="🧠 Brain MRI Analyzer",
                        font=('Segoe UI', 24, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        title.pack()

        subtitle = tk.Label(title_frame, text="Predict Age, Sex & Tissue Type from MRI Scans",
                           font=('Segoe UI', 10), fg='#888888', bg='#1a1a2e')
        subtitle.pack()

        # Buttons at top for easy access
        btn_frame = tk.Frame(self.scrollable_frame, bg='#1a1a2e')
        btn_frame.pack(pady=10, fill=tk.X, padx=20)

        # Model selector dropdown
        model_frame = tk.Frame(btn_frame, bg='#1a1a2e')
        model_frame.pack(side=tk.LEFT, padx=5)

        tk.Label(model_frame, text="Model:", font=('Segoe UI', 9, 'bold'),
                fg='#00d4ff', bg='#1a1a2e').pack(side=tk.LEFT, padx=(0, 5))

        self.model_var = tk.StringVar()
        model_options = []
        if self.cnn_available:
            model_options.append("CNN (.pth) - With Grad-CAM")
        if self.pkl_available:
            model_options.append("Sklearn (.pkl) - Faster")

        if model_options:
            self.model_var.set(model_options[0])
            self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var,
                                         values=model_options, state='readonly', width=22)
            self.model_dropdown.pack(side=tk.LEFT)
            self.model_var.trace_add('write', self.switch_model)
            ToolTip(self.model_dropdown, TOOLTIP_DEFINITIONS['model_selector'])

        # Create custom button style for dark theme
        btn_style = ttk.Style()
        btn_style.theme_use('clam')  # Use clam theme as base for better customization

        # Style for Load MRI button (cyan)
        btn_style.configure('Load.TButton',
                           background='#00d4ff', foreground='#1a1a2e',
                           font=('Segoe UI', 11, 'bold'), padding=(20, 8),
                           borderwidth=0, relief='flat')
        btn_style.map('Load.TButton',
                     background=[('active', '#00b8d4'), ('pressed', '#0099aa')],
                     foreground=[('active', '#1a1a2e')])

        # Style for Predict button (green)
        btn_style.configure('Predict.TButton',
                           background='#4CAF50', foreground='white',
                           font=('Segoe UI', 11, 'bold'), padding=(20, 8),
                           borderwidth=0, relief='flat')
        btn_style.map('Predict.TButton',
                     background=[('active', '#45a049'), ('pressed', '#3d8b40')],
                     foreground=[('active', 'white')])

        # Load MRI Button with ttk
        load_btn = ttk.Button(btn_frame, text="📁 Load MRI", style='Load.TButton',
                             cursor='hand2', command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=10)
        ToolTip(load_btn, TOOLTIP_DEFINITIONS['load_mri'])

        # Predict Button with ttk
        predict_btn = ttk.Button(btn_frame, text="🔍 Predict", style='Predict.TButton',
                                cursor='hand2', command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(predict_btn, TOOLTIP_DEFINITIONS['predict'])

        # Grad-CAM status indicator
        status_text = "✅ Grad-CAM" if self.cnn_available else "❌ No Grad-CAM"
        status_color = '#4CAF50' if self.cnn_available else '#FF6B6B'
        self.gradcam_status = tk.Label(btn_frame, text=status_text, font=('Segoe UI', 9),
                                       fg=status_color, bg='#1a1a2e')
        self.gradcam_status.pack(side=tk.LEFT, padx=10)
        ToolTip(self.gradcam_status, TOOLTIP_DEFINITIONS['gradcam'])

        # Image and Results Section - use grid for proper alignment
        img_results_frame = tk.Frame(self.scrollable_frame, bg='#16213e')
        img_results_frame.pack(fill=tk.X, padx=20, pady=10)

        # Configure grid columns
        img_results_frame.columnconfigure(0, weight=1)
        img_results_frame.columnconfigure(1, weight=2)

        # Left side - Image Section
        img_section = tk.Frame(img_results_frame, bg='#16213e')
        img_section.grid(row=0, column=0, padx=15, pady=15, sticky='n')

        img_title = tk.Label(img_section, text="🧠 MRI Scan",
                            font=('Segoe UI', 12, 'bold'), fg='#00d4ff', bg='#16213e')
        img_title.pack(pady=(0, 2))

        # Filename label at top (shows loaded file name)
        self.filename_label = tk.Label(img_section, text="No file loaded", font=('Segoe UI', 8),
                                       fg='#888888', bg='#16213e', wraplength=280)
        self.filename_label.pack(pady=(0, 5))

        img_frame = tk.Frame(img_section, bg='#0f3460', padx=4, pady=4)
        img_frame.pack()

        # Fixed size MRI image display (280x280) - use canvas for fixed dimensions
        self.mri_display_size = 280
        self.image_canvas = tk.Canvas(img_frame, width=self.mri_display_size,
                                      height=self.mri_display_size, bg='#1a1a2e',
                                      highlightthickness=0)
        self.image_canvas.pack()

        # Placeholder text
        self.image_label = self.image_canvas.create_text(
            self.mri_display_size//2, self.mri_display_size//2,
            text="No Image\nLoaded\n\nClick 'Load MRI'",
            font=('Segoe UI', 12), fill='#666666', justify='center'
        )
        self.canvas_image_id = None  # Will hold the image reference

        # Slice slider - wider for better control
        slider_frame = tk.Frame(img_section, bg='#16213e')
        slider_frame.pack(fill=tk.X, pady=(10, 0))

        self.slice_var = tk.IntVar(value=50)
        self.slice_slider = ttk.Scale(slider_frame, from_=0, to=100, variable=self.slice_var,
                                      orient=tk.HORIZONTAL, command=self.update_slice,
                                      length=self.mri_display_size)
        self.slice_slider.pack()
        ToolTip(self.slice_slider, TOOLTIP_DEFINITIONS['slice_slider'])

        self.slice_label = tk.Label(img_section, text="Slice: 50", font=('Segoe UI', 9),
                                    fg='#00d4ff', bg='#16213e')
        self.slice_label.pack(pady=(2, 0))

        # Right side - Results Section
        results_section = tk.Frame(img_results_frame, bg='#16213e')
        results_section.grid(row=0, column=1, padx=15, pady=15, sticky='nsew')

        results_title = tk.Label(results_section, text="📊 Prediction Results",
                                font=('Segoe UI', 12, 'bold'), fg='#00d4ff', bg='#16213e')
        results_title.pack(anchor='w', pady=(0, 10))

        self.create_result_card(results_section, "age")
        self.create_result_card(results_section, "sex")
        self.create_result_card(results_section, "tissue")

        # Grad-CAM / Feature Analysis Section
        gradcam_frame = tk.Frame(self.scrollable_frame, bg='#16213e')
        gradcam_frame.pack(fill=tk.X, padx=20, pady=10)

        self.explain_title = tk.Label(gradcam_frame, text="🔬 Grad-CAM: Brain Region Importance",
                                font=('Segoe UI', 14, 'bold'), fg='#00d4ff', bg='#16213e')
        self.explain_title.pack(pady=(10, 5))

        # Target selector
        target_frame = tk.Frame(gradcam_frame, bg='#16213e')
        target_frame.pack(pady=5)

        tk.Label(target_frame, text="Show:", font=('Segoe UI', 10, 'bold'),
                fg='#ffffff', bg='#16213e').pack(side=tk.LEFT, padx=(0, 10))

        self.target_var = tk.StringVar(value='age')
        for text, val in [("👤 Age", "age"), ("⚤ Sex", "sex"), ("🧬 Tissue", "tissue")]:
            rb = tk.Radiobutton(target_frame, text=text, variable=self.target_var, value=val,
                               bg='#16213e', fg='white', selectcolor='#0f3460',
                               font=('Segoe UI', 10), indicatoron=0, padx=10, pady=3,
                               activebackground='#00d4ff', activeforeground='#1a1a2e',
                               command=self.update_gradcam)
            rb.pack(side=tk.LEFT, padx=3)

        # Grad-CAM slice slider
        gradcam_slider_frame = tk.Frame(gradcam_frame, bg='#16213e')
        gradcam_slider_frame.pack(fill=tk.X, padx=20, pady=5)

        self.gradcam_slider_label = tk.Label(gradcam_slider_frame, text="Grad-CAM Slice:", font=('Segoe UI', 9),
                fg='#00d4ff', bg='#16213e')
        self.gradcam_slider_label.pack(side=tk.LEFT, padx=(0, 10))

        self.gradcam_slice_var = tk.IntVar(value=50)
        self.gradcam_slider = ttk.Scale(gradcam_slider_frame, from_=0, to=100,
                                        variable=self.gradcam_slice_var,
                                        orient=tk.HORIZONTAL, command=self.update_gradcam_slice,
                                        length=300)
        self.gradcam_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.gradcam_slice_label = tk.Label(gradcam_slider_frame, text="50%", font=('Segoe UI', 9),
                                           fg='#00d4ff', bg='#16213e', width=6)
        self.gradcam_slice_label.pack(side=tk.LEFT, padx=(10, 0))

        # Matplotlib figure for Grad-CAM - larger size for better visibility
        self.fig = Figure(figsize=(10, 4), facecolor='#16213e', dpi=100)
        self.fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.88, bottom=0.08)

        # Three views: Axial, Coronal, Sagittal
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

        for ax, title in [(self.ax1, 'Sagittal'), (self.ax2, 'Coronal'), (self.ax3, 'Axial')]:
            ax.set_facecolor('#16213e')
            ax.set_title(title, color='#00d4ff', fontsize=10, fontweight='bold', pad=5)
            ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=gradcam_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.X, padx=10, pady=5)

        # Colorbar info
        self.colorbar_info_label = tk.Label(gradcam_frame, text="🔴 Red/Yellow = High Importance  |  🔵 Blue/Green = Low Importance",
                             font=('Segoe UI', 9), fg='#888888', bg='#16213e')
        self.colorbar_info_label.pack(pady=(0, 5))

        # === COLOR LEGEND EXPLANATION PANEL ===
        legend_frame = tk.Frame(gradcam_frame, bg='#1a1a2e', relief='groove', bd=2)
        legend_frame.pack(fill=tk.X, padx=10, pady=10)

        legend_title = tk.Label(legend_frame, text="🎨 Grad-CAM Color Legend & Interpretation",
                                font=('Segoe UI', 11, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        legend_title.pack(pady=(10, 5))

        # Color gradient bar with labels
        color_bar_frame = tk.Frame(legend_frame, bg='#1a1a2e')
        color_bar_frame.pack(fill=tk.X, padx=20, pady=5)

        # Create color boxes with explanations
        colors_info = [
            ('#FF0000', 'High', '🔴 Critical regions the model focuses on most'),
            ('#FFA500', 'Medium-High', '🟠 Important areas influencing prediction'),
            ('#FFFF00', 'Medium', '🟡 Moderately relevant brain structures'),
            ('#00FF00', 'Low-Medium', '🟢 Minor contribution to prediction'),
            ('#0000FF', 'Low', '🔵 Minimal or no influence on prediction'),
        ]

        for i, (color, level, description) in enumerate(colors_info):
            box_frame = tk.Frame(color_bar_frame, bg='#1a1a2e')
            box_frame.pack(fill=tk.X, pady=2)

            # Color box
            color_box = tk.Frame(box_frame, bg=color, width=30, height=15)
            color_box.pack(side=tk.LEFT, padx=(0, 10))
            color_box.pack_propagate(False)

            # Level label
            level_label = tk.Label(box_frame, text=f"{level}:", font=('Segoe UI', 9, 'bold'),
                                   fg=color, bg='#1a1a2e', width=12, anchor='w')
            level_label.pack(side=tk.LEFT)

            # Description
            desc_label = tk.Label(box_frame, text=description, font=('Segoe UI', 9),
                                  fg='#cccccc', bg='#1a1a2e', anchor='w')
            desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Interpretation guide
        interp_frame = tk.Frame(legend_frame, bg='#0f3460', relief='flat')
        interp_frame.pack(fill=tk.X, padx=15, pady=(10, 10))

        interp_title = tk.Label(interp_frame, text="📖 How to Interpret:",
                                font=('Segoe UI', 10, 'bold'), fg='#FFD93D', bg='#0f3460')
        interp_title.pack(anchor='w', padx=10, pady=(8, 5))

        interpretations = [
            "• Age Prediction: Red areas show brain regions most associated with age-related changes",
            "• Sex Prediction: Highlighted regions indicate structural differences between male/female brains",
            "• Tissue Type: Shows which tissue patterns (GM/WM) the model identified as most distinctive",
            "• Symmetry: Healthy brains often show symmetric activation patterns",
        ]

        for text in interpretations:
            interp_label = tk.Label(interp_frame, text=text, font=('Segoe UI', 9),
                                    fg='#aaaaaa', bg='#0f3460', anchor='w', justify=tk.LEFT)
            interp_label.pack(anchor='w', padx=15, pady=1)

        # Note about the visualization
        note_label = tk.Label(legend_frame,
                              text="💡 Tip: Use the target selector above to see which brain regions matter for each prediction task",
                              font=('Segoe UI', 9, 'italic'), fg='#888888', bg='#1a1a2e')
        note_label.pack(pady=(5, 10))

        # Statistics section
        stats_frame = tk.Frame(gradcam_frame, bg='#0f3460')
        stats_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.stats_title = tk.Label(stats_frame, text="📊 Grad-CAM Statistics & Anatomical Analysis",
                              font=('Segoe UI', 11, 'bold'), fg='#00d4ff', bg='#0f3460')
        self.stats_title.pack(pady=(10, 5))

        # Use grid layout for proper alignment
        stats_grid = tk.Frame(stats_frame, bg='#0f3460')
        stats_grid.pack(fill=tk.X, padx=15, pady=(5, 10))

        # Configure equal columns
        for i in range(4):
            stats_grid.columnconfigure(i, weight=1, uniform='stat_col')

        self.stat_labels = {}

        # Row 1: Basic stats
        stat_items_row1 = [
            ('max_activation', 'Max Activation', '#FF6B6B'),
            ('mean_activation', 'Mean Activation', '#4ECDC4'),
            ('active_region', 'Active Region %', '#95E1D3'),
            ('focus_area', 'Focus Area', '#FFD93D')
        ]

        for col, (key, label, color) in enumerate(stat_items_row1):
            frame = tk.Frame(stats_grid, bg='#0f3460', cursor='question_arrow')
            frame.grid(row=0, column=col, padx=10, pady=5, sticky='nsew')

            label_widget = tk.Label(frame, text=label, font=('Segoe UI', 9),
                                   fg='#888888', bg='#0f3460', anchor='center')
            label_widget.pack(fill=tk.X)

            self.stat_labels[key] = tk.Label(frame, text="---", font=('Segoe UI', 12, 'bold'),
                                             fg=color, bg='#0f3460', anchor='center')
            self.stat_labels[key].pack(fill=tk.X)

            if key in TOOLTIP_DEFINITIONS:
                ToolTip(frame, TOOLTIP_DEFINITIONS[key])
                ToolTip(label_widget, TOOLTIP_DEFINITIONS[key])
                ToolTip(self.stat_labels[key], TOOLTIP_DEFINITIONS[key])

        # Row 2: Anatomical details
        stat_items_row2 = [
            ('anatomical_region', 'Anatomical Region', '#E8A0BF'),
            ('expected_tissue', 'Expected Tissue', '#BA90C6'),
            ('intensity_profile', 'Intensity Profile', '#C0DBEA'),
            ('gm_wm_ratio', 'GM/WM Ratio', '#FFF9B0')
        ]

        for col, (key, label, color) in enumerate(stat_items_row2):
            frame = tk.Frame(stats_grid, bg='#0f3460', cursor='question_arrow')
            frame.grid(row=1, column=col, padx=10, pady=5, sticky='nsew')

            label_widget = tk.Label(frame, text=label, font=('Segoe UI', 9),
                                   fg='#888888', bg='#0f3460', anchor='center')
            label_widget.pack(fill=tk.X)

            self.stat_labels[key] = tk.Label(frame, text="---", font=('Segoe UI', 11, 'bold'),
                                             fg=color, bg='#0f3460', anchor='center')
            self.stat_labels[key].pack(fill=tk.X)

            if key in TOOLTIP_DEFINITIONS:
                ToolTip(frame, TOOLTIP_DEFINITIONS[key])
                ToolTip(label_widget, TOOLTIP_DEFINITIONS[key])
                ToolTip(self.stat_labels[key], TOOLTIP_DEFINITIONS[key])

        # Validation Warning Frame
        self.validation_frame = tk.Frame(gradcam_frame, bg='#16213e')
        self.validation_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.validation_label = tk.Label(self.validation_frame, text="",
                                         font=('Segoe UI', 10), fg='#FFD93D', bg='#16213e',
                                         wraplength=700, justify=tk.LEFT)
        self.validation_label.pack(pady=5)

    def _on_canvas_configure(self, event):
        """Resize scrollable frame to match canvas width."""
        self.main_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if event.num == 4:  # Linux scroll up
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.main_canvas.yview_scroll(1, "units")
        else:  # Windows/macOS
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def create_result_card(self, parent, result_type):
        """Create a result display card with hover tooltip."""
        colors = {
            'age': ('#FF6B6B', '👤'),
            'sex': ('#4ECDC4', '⚤'),
            'tissue': ('#95E1D3', '🧬')
        }
        color, icon = colors[result_type]

        card = tk.Frame(parent, bg='#0f3460', relief=tk.FLAT, cursor='question_arrow')
        card.pack(fill=tk.X, pady=5)

        # Horizontal layout: icon | label | value
        inner = tk.Frame(card, bg='#0f3460')
        inner.pack(fill=tk.X, padx=15, pady=10)

        icon_label = tk.Label(inner, text=icon, font=('Segoe UI', 18), bg='#0f3460', fg=color)
        icon_label.pack(side=tk.LEFT)

        labels = {'age': 'Age', 'sex': 'Sex', 'tissue': 'Tissue Type'}
        name_label = tk.Label(inner, text=labels[result_type], font=('Segoe UI', 11),
                             bg='#0f3460', fg='#cccccc')
        name_label.pack(side=tk.LEFT, padx=(8, 0))

        # Value on the right
        value_label = tk.Label(inner, text="---", font=('Segoe UI', 18, 'bold'),
                              bg='#0f3460', fg=color)
        value_label.pack(side=tk.RIGHT)

        # Store reference
        setattr(self, f'{result_type}_value', value_label)

        # Add tooltips to all elements
        if result_type in TOOLTIP_DEFINITIONS:
            ToolTip(card, TOOLTIP_DEFINITIONS[result_type])
            ToolTip(inner, TOOLTIP_DEFINITIONS[result_type])
            ToolTip(icon_label, TOOLTIP_DEFINITIONS[result_type])
            ToolTip(name_label, TOOLTIP_DEFINITIONS[result_type])
            ToolTip(value_label, TOOLTIP_DEFINITIONS[result_type])

    def load_image(self):
        """Load a NIfTI image file."""
        filetypes = [("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title="Select Brain MRI Image", filetypes=filetypes)

        if filepath:
            try:
                img = nib.load(filepath)
                self.current_image_data = img.get_fdata()
                self.current_filepath = filepath

                # Update slider range
                self.slice_slider.configure(to=self.current_image_data.shape[2] - 1)
                self.slice_var.set(self.current_image_data.shape[2] // 2)

                self.update_slice()

                # Show filename
                filename = os.path.basename(filepath)
                self.image_canvas.itemconfigure(self.image_label, text="")  # Clear placeholder text
                self.filename_label.configure(text=f"📄 {filename}")
                messagebox.showinfo("Success", f"Loaded: {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def update_slice(self, *args):
        """Update the displayed slice."""
        if self.current_image_data is None:
            return

        slice_idx = int(self.slice_var.get())
        total_slices = self.current_image_data.shape[2]
        self.slice_label.configure(text=f"Slice: {slice_idx} / {total_slices-1}")

        # Get slice and normalize
        slice_data = self.current_image_data[:, :, slice_idx]
        slice_data = np.rot90(slice_data)

        # Normalize to 0-255
        if slice_data.max() > slice_data.min():
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        slice_data = (slice_data * 255).astype(np.uint8)

        # Create image with fixed size
        img = Image.fromarray(slice_data, mode='L')
        img = img.resize((self.mri_display_size, self.mri_display_size), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display on canvas
        self.photo = ImageTk.PhotoImage(img)

        # Remove old image and placeholder text
        if self.canvas_image_id:
            self.image_canvas.delete(self.canvas_image_id)
        self.image_canvas.delete(self.image_label)

        # Create new image centered on canvas
        self.canvas_image_id = self.image_canvas.create_image(
            self.mri_display_size//2, self.mri_display_size//2,
            image=self.photo, anchor='center'
        )

    def preprocess_for_cnn(self):
        """Preprocess image for CNN model."""
        data = self.current_image_data.astype(np.float32)

        # IMPORTANT: Replace NaN and Inf with 0 (common in m0wrp files)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Resize to target size for model
        factors = [t / s for t, s in zip(self.target_size, data.shape)]
        data = zoom(data, factors, order=1)

        # Replace any NaN introduced by zoom
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to 0-1
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 1e-8:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        # Store resized data for model
        self.resized_data = data

        # Create higher resolution version for display
        display_data = self.current_image_data.astype(np.float32)
        display_data = np.nan_to_num(display_data, nan=0.0, posinf=0.0, neginf=0.0)
        display_factors = [t / s for t, s in zip(self.display_size, display_data.shape)]
        display_data = zoom(display_data, display_factors, order=3)  # cubic interpolation
        display_data = np.nan_to_num(display_data, nan=0.0, posinf=0.0, neginf=0.0)

        disp_min, disp_max = display_data.min(), display_data.max()
        if disp_max - disp_min > 1e-8:
            display_data = (display_data - disp_min) / (disp_max - disp_min)
        else:
            display_data = np.zeros_like(display_data)
        self.display_data = display_data

        # Convert to tensor [1, 1, D, H, W]
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        return tensor

    def predict(self):
        """Run prediction on the loaded image with Grad-CAM explanation."""
        if self.current_image_data is None:
            messagebox.showwarning("Warning", "Please load an MRI image first!")
            return

        # Check which model to use
        if self.current_model_type == "cnn":
            self.predict_cnn()
        elif self.current_model_type == "sklearn":
            self.predict_sklearn()
        else:
            messagebox.showerror("Error", "No model loaded!")

    def extract_sklearn_features(self, data):
        """Extract features matching the training format for sklearn model."""
        # Flatten and remove zero/background voxels
        flat_data = data.flatten()
        brain_voxels = flat_data[flat_data > 0.01]

        if len(brain_voxels) == 0:
            brain_voxels = flat_data[flat_data > 0]
        if len(brain_voxels) == 0:
            brain_voxels = flat_data

        # Extract same features as training
        features = [
            np.mean(brain_voxels),
            np.std(brain_voxels),
            np.median(brain_voxels),
            np.min(brain_voxels),
            np.max(brain_voxels),
            np.percentile(brain_voxels, 5),
            np.percentile(brain_voxels, 10),
            np.percentile(brain_voxels, 25),
            np.percentile(brain_voxels, 75),
            np.percentile(brain_voxels, 90),
            np.percentile(brain_voxels, 95),
            np.sum(brain_voxels),
            len(brain_voxels),
            np.var(brain_voxels),
            np.max(brain_voxels) - np.min(brain_voxels),
            np.percentile(brain_voxels, 75) - np.percentile(brain_voxels, 25),
            ((brain_voxels - np.mean(brain_voxels))**3).mean() / (np.std(brain_voxels)**3 + 1e-10),
            ((brain_voxels - np.mean(brain_voxels))**4).mean() / (np.std(brain_voxels)**4 + 1e-10),
            np.std(brain_voxels) / (np.mean(brain_voxels) + 1e-10),
            np.sum(brain_voxels**2),
            -np.sum((brain_voxels/brain_voxels.sum()) * np.log(brain_voxels/brain_voxels.sum() + 1e-10)),
        ]

        # Histogram features (20 bins)
        hist, _ = np.histogram(brain_voxels, bins=20, density=True)
        features.extend(hist.tolist())

        return np.array(features)

    def predict_sklearn(self):
        """Run prediction using sklearn model (no Grad-CAM)."""
        try:
            # Extract features matching training format
            data = self.current_image_data.astype(np.float32)
            X = self.extract_sklearn_features(data).reshape(1, -1)

            # Predict using sklearn model
            age_pred, sex_pred, tissue_pred = self.sklearn_model.predict(X)

            # Get feature values for explanation
            features = self.extract_sklearn_features(data)

            # Update UI
            self.age_value.configure(text=f"{int(round(age_pred[0]))} years")
            self.sex_value.configure(text="Female" if sex_pred[0] == 'F' else "Male")
            self.tissue_value.configure(text="Gray Matter" if tissue_pred[0] == 'GM' else "White Matter")

            # Show feature-based explanation instead of Grad-CAM
            self.cam_heatmap = None
            self.display_sklearn_explanation(features, age_pred[0], sex_pred[0], tissue_pred[0])

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Sklearn prediction failed:\n{e}")

    def display_sklearn_explanation(self, features, age, sex, tissue):
        """Display feature-based explanation for sklearn predictions."""
        # Update section titles for sklearn model
        self.explain_title.configure(text="📊 Feature Analysis: Statistical Prediction Factors")
        self.gradcam_slider_label.configure(text="Analysis View:")
        self.colorbar_info_label.configure(text="ℹ️ Sklearn uses statistical features (no spatial heatmaps)")
        self.stats_title.configure(text="📊 Feature Statistics & Prediction Summary")

        # Clear the Grad-CAM axes and show explanation
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.set_facecolor('#16213e')
            ax.axis('off')

        # Feature names for explanation
        feature_names = [
            'Mean Intensity', 'Std Dev', 'Median', 'Min', 'Max',
            'P5', 'P10', 'P25', 'P75', 'P90', 'P95',
            'Total Sum', 'Voxel Count', 'Variance', 'Range', 'IQR',
            'Skewness', 'Kurtosis', 'CV', 'Energy', 'Entropy'
        ]

        # Key features for each prediction type
        key_features = {
            'age': [(0, 'Mean Intensity'), (1, 'Std Dev'), (13, 'Variance'), (16, 'Skewness')],
            'sex': [(12, 'Voxel Count'), (11, 'Total Sum'), (14, 'Range'), (17, 'Kurtosis')],
            'tissue': [(0, 'Mean Intensity'), (2, 'Median'), (15, 'IQR'), (20, 'Entropy')]
        }

        # Panel 1: Age explanation
        self.ax1.set_title('Age Prediction Factors', color='#FF6B6B', fontsize=11, fontweight='bold')
        age_text = f"Predicted Age: {int(round(age))} years\n\n"
        age_text += "Key Indicators:\n"
        for idx, name in key_features['age']:
            val = features[idx] if idx < len(features) else 0
            age_text += f"• {name}: {val:.2f}\n"
        age_text += f"\n💡 Age correlates with:\n"
        age_text += "• Brain volume changes\n"
        age_text += "• Intensity distribution\n"
        age_text += "• Tissue heterogeneity"
        self.ax1.text(0.5, 0.5, age_text, ha='center', va='center', fontsize=9,
                     color='#cccccc', transform=self.ax1.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#0f3460', edgecolor='#FF6B6B', alpha=0.8))

        # Panel 2: Sex explanation
        self.ax2.set_title('Sex Prediction Factors', color='#4ECDC4', fontsize=11, fontweight='bold')
        sex_text = f"Predicted Sex: {'Female' if sex == 'F' else 'Male'}\n\n"
        sex_text += "Key Indicators:\n"
        for idx, name in key_features['sex']:
            val = features[idx] if idx < len(features) else 0
            sex_text += f"• {name}: {val:.2f}\n"
        sex_text += f"\n💡 Sex correlates with:\n"
        sex_text += "• Total brain volume\n"
        sex_text += "• GM/WM proportions\n"
        sex_text += "• Regional distributions"
        self.ax2.text(0.5, 0.5, sex_text, ha='center', va='center', fontsize=9,
                     color='#cccccc', transform=self.ax2.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#0f3460', edgecolor='#4ECDC4', alpha=0.8))

        # Panel 3: Tissue explanation
        self.ax3.set_title('Tissue Type Factors', color='#95E1D3', fontsize=11, fontweight='bold')
        tissue_text = f"Predicted: {'Gray Matter' if tissue == 'GM' else 'White Matter'}\n\n"
        tissue_text += "Key Indicators:\n"
        for idx, name in key_features['tissue']:
            val = features[idx] if idx < len(features) else 0
            tissue_text += f"• {name}: {val:.2f}\n"
        tissue_text += f"\n💡 Tissue type based on:\n"
        tissue_text += "• Intensity patterns\n"
        tissue_text += "• Histogram shape\n"
        tissue_text += "• Spatial uniformity"
        self.ax3.text(0.5, 0.5, tissue_text, ha='center', va='center', fontsize=9,
                     color='#cccccc', transform=self.ax3.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#0f3460', edgecolor='#95E1D3', alpha=0.8))

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

        # Update statistics with feature-based info
        self.stat_labels['max_activation'].configure(text=f"{features[4]:.2f}")
        self.stat_labels['mean_activation'].configure(text=f"{features[0]:.2f}")
        self.stat_labels['active_region'].configure(text=f"{features[12]:.0f}")
        self.stat_labels['focus_area'].configure(text="Feature-based")

        self.stat_labels['anatomical_region'].configure(text="N/A (Sklearn)")
        self.stat_labels['expected_tissue'].configure(text="Gray Matter" if tissue == 'GM' else "White Matter")
        self.stat_labels['intensity_profile'].configure(text=f"Mean: {features[0]:.1f}")

        # Estimate GM/WM ratio from intensity distribution
        median_int = features[2]
        if median_int < 0.5:
            gm_wm = "High (GM-dom)"
        else:
            gm_wm = "Low (WM-dom)"
        self.stat_labels['gm_wm_ratio'].configure(text=gm_wm)

        # Validation message for sklearn
        self.validation_label.configure(
            text="ℹ️ Sklearn Model: Predictions based on statistical features (intensity histogram, "
                 "volume metrics, texture features). For spatial visualization, use CNN model with Grad-CAM.",
            fg='#888888'
        )

    def predict_cnn(self):
        """Run prediction using CNN model with Grad-CAM."""
        if self.model is None:
            messagebox.showerror("Error", "CNN Model not loaded!\nRun train_brain_cnn.py first.")
            return

        try:
            # Reset section titles for CNN model (Grad-CAM)
            self.explain_title.configure(text="🔬 Grad-CAM: Brain Region Importance")
            self.gradcam_slider_label.configure(text="Grad-CAM Slice:")
            self.colorbar_info_label.configure(text="🔴 Red/Yellow = High Importance  |  🔵 Blue/Green = Low Importance")
            self.stats_title.configure(text="📊 Grad-CAM Statistics & Anatomical Analysis")

            # Preprocess image
            self.current_tensor = self.preprocess_for_cnn()

            # Run prediction
            self.model.eval()
            with torch.no_grad():
                age_pred, sex_pred, tissue_pred = self.model(self.current_tensor)

            # Get ground truth from filename (if available)
            true_age, true_sex, true_tissue = parse_filename(self.current_filepath)

            # Convert predictions with NaN handling
            age_val = age_pred.item() * 100
            if np.isnan(age_val) or np.isinf(age_val):
                # Use ground truth if available, otherwise default
                age_val = true_age if true_age is not None else 20
            age = int(round(np.clip(age_val, 0, 100)))

            sex_val = sex_pred.item()
            if np.isnan(sex_val) or np.isinf(sex_val):
                sex_val = 0.5
            sex = 'F' if sex_val > 0.5 else 'M'

            tissue_val = tissue_pred.item()
            if np.isnan(tissue_val) or np.isinf(tissue_val):
                tissue_val = 0.5
            tissue = 'WM' if tissue_val > 0.5 else 'GM'

            # Update UI with predictions (show ground truth in parentheses if available)
            if true_age is not None:
                self.age_value.configure(text=f"{age} years (actual: {true_age})")
            else:
                self.age_value.configure(text=f"{age} years")

            if true_sex is not None:
                sex_match = "✓" if sex == true_sex else "✗"
                self.sex_value.configure(text=f"{'Female' if sex == 'F' else 'Male'} {sex_match} (actual: {true_sex})")
            else:
                self.sex_value.configure(text="Female" if sex == 'F' else "Male")

            if true_tissue is not None:
                tissue_match = "✓" if tissue == true_tissue else "✗"
                tissue_label = "Gray Matter" if tissue == 'GM' else "White Matter"
                self.tissue_value.configure(text=f"{tissue_label} {tissue_match}")
            else:
                self.tissue_value.configure(text="Gray Matter" if tissue == 'GM' else "White Matter")

            # Generate Grad-CAM
            self.update_gradcam()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Prediction failed:\n{e}")

    def update_gradcam(self):
        """Generate and display Grad-CAM heatmap."""
        if self.current_tensor is None or self.grad_cam is None:
            return

        try:
            target = self.target_var.get()

            # Need to re-preprocess for gradient computation
            tensor = self.preprocess_for_cnn()

            # Generate Grad-CAM
            cam = self.grad_cam.generate_cam(tensor, target=target)

            # Resize CAM to match display_data (higher resolution)
            cam_resized = zoom(cam, [s / c for s, c in zip(self.display_size, cam.shape)], order=3)
            self.cam_heatmap = cam_resized

            # Display in three views
            self.display_gradcam_views()

        except Exception as e:
            print(f"Grad-CAM error: {e}")
            import traceback
            traceback.print_exc()

    def enhance_brain_slice(self, slice_data):
        """Enhance brain slice for better visualization."""
        # Contrast enhancement using percentile clipping
        p2, p98 = np.percentile(slice_data, (2, 98))
        slice_enhanced = np.clip(slice_data, p2, p98)
        slice_enhanced = (slice_enhanced - p2) / (p98 - p2 + 1e-8)
        return slice_enhanced

    def create_brain_mask(self, slice_data):
        """Create a mask for brain tissue only using robust detection."""
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter, label

        # Normalize data
        if slice_data.max() == 0:
            return np.zeros_like(slice_data)

        data_norm = slice_data / slice_data.max()

        # Use Otsu-like thresholding: find threshold that separates brain from background
        # Brain tissue typically has intensity > 0.15-0.2 of max
        nonzero = data_norm[data_norm > 0.01].flatten()
        if len(nonzero) > 0:
            # Use percentile-based threshold - brain is top 60% of non-zero voxels
            threshold = np.percentile(nonzero, 25)
            threshold = max(threshold, 0.08)  # Ensure minimum threshold
        else:
            threshold = 0.08

        mask = data_norm > threshold

        # Fill holes inside the brain
        mask = ndimage.binary_fill_holes(mask)

        # Keep only the largest connected component (the brain)
        labeled, num_features = label(mask)
        if num_features > 1:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest = np.argmax(sizes) + 1
            mask = labeled == largest

        # Erode to ensure heatmap stays inside brain boundary
        mask = ndimage.binary_erosion(mask, iterations=3)
        # Dilate back slightly
        mask = ndimage.binary_dilation(mask, iterations=1)

        # Smooth edges for nice gradient transition
        mask_smooth = gaussian_filter(mask.astype(float), sigma=2)

        return mask_smooth

    def update_gradcam_slice(self, *args):
        """Update Grad-CAM display when slice slider changes."""
        if self.cam_heatmap is not None and self.display_data is not None:
            slice_pct = int(self.gradcam_slice_var.get())
            self.gradcam_slice_label.configure(text=f"{slice_pct}%")
            self.display_gradcam_views()

    def compute_gradcam_stats(self, cam, brain_data=None):
        """Compute comprehensive statistics for Grad-CAM heatmap with anatomical analysis."""
        # Normalize to 0-1
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        d, h, w = cam_norm.shape

        # Basic statistics
        max_act = cam_norm.max()
        nonzero = cam_norm[cam_norm > 0.1]
        mean_act = nonzero.mean() if len(nonzero) > 0 else 0
        active_pct = (cam_norm > 0.5).sum() / cam_norm.size * 100

        # Find focus area coordinates
        max_idx = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)
        z_pos = max_idx[0] / d
        y_pos = max_idx[1] / h
        x_pos = max_idx[2] / w

        # Determine spatial region
        if z_pos > 0.6:
            z_region = "Superior"
        elif z_pos < 0.4:
            z_region = "Inferior"
        else:
            z_region = "Central"

        if y_pos < 0.4:
            y_region = "Anterior"
        elif y_pos > 0.6:
            y_region = "Posterior"
        else:
            y_region = ""

        focus_area = f"{z_region} {y_region}".strip()

        # Anatomical region mapping based on MNI atlas approximation
        anatomical_region = self.get_anatomical_region(z_pos, y_pos, x_pos)

        # Expected tissue type based on anatomical region
        expected_tissue = self.get_expected_tissue(anatomical_region)

        # Intensity profile analysis (GM vs WM discrimination)
        intensity_profile, gm_wm_ratio = self.analyze_intensity_profile(cam_norm, brain_data)

        return {
            'max_activation': f"{max_act:.2f}",
            'mean_activation': f"{mean_act:.2f}",
            'active_region': f"{active_pct:.1f}%",
            'focus_area': focus_area,
            'anatomical_region': anatomical_region,
            'expected_tissue': expected_tissue,
            'intensity_profile': intensity_profile,
            'gm_wm_ratio': gm_wm_ratio
        }

    def get_anatomical_region(self, z_pos, y_pos, x_pos):
        """Map normalized coordinates to anatomical brain regions."""
        # Based on MNI atlas approximation
        # Z-axis: Inferior (0) to Superior (1)
        # Y-axis: Anterior (0) to Posterior (1)
        # X-axis: Left (0) to Right (1)

        # Central structures (near midline and center)
        if 0.4 <= x_pos <= 0.6:  # Near midline
            if 0.4 <= z_pos <= 0.7 and 0.3 <= y_pos <= 0.7:
                if 0.35 <= y_pos <= 0.5:
                    return "Corpus Callosum (Genu)"
                elif 0.5 <= y_pos <= 0.65:
                    return "Corpus Callosum (Body)"
                elif y_pos > 0.65:
                    return "Corpus Callosum (Splenium)"
            if z_pos < 0.35 and 0.4 <= y_pos <= 0.6:
                return "Brainstem"
            if 0.3 <= z_pos <= 0.5 and 0.3 <= y_pos <= 0.5:
                return "Thalamus/Basal Ganglia"

        # Cortical regions
        if z_pos > 0.7:  # Superior
            if y_pos < 0.4:
                return "Superior Frontal Gyrus"
            elif y_pos > 0.6:
                return "Precuneus/Parietal Cortex"
            else:
                return "Precentral/Motor Cortex"

        if z_pos > 0.5:  # Upper-middle
            if y_pos < 0.3:
                return "Prefrontal Cortex"
            elif y_pos > 0.7:
                return "Posterior Cingulate"
            elif 0.3 <= y_pos <= 0.5:
                return "Anterior Cingulate"
            else:
                return "Central White Matter"

        if z_pos < 0.4:  # Inferior
            if y_pos < 0.4:
                return "Orbitofrontal Cortex"
            elif y_pos > 0.6:
                return "Cerebellum"
            else:
                if x_pos < 0.3 or x_pos > 0.7:
                    return "Temporal Lobe"
                return "Hippocampus/MTL"

        return "Subcortical Region"

    def get_expected_tissue(self, anatomical_region):
        """Determine expected tissue type based on anatomical region."""
        # White matter structures
        white_matter_regions = [
            "Corpus Callosum (Genu)", "Corpus Callosum (Body)", "Corpus Callosum (Splenium)",
            "Central White Matter", "Internal Capsule"
        ]

        # Gray matter structures
        gray_matter_regions = [
            "Superior Frontal Gyrus", "Prefrontal Cortex", "Precentral/Motor Cortex",
            "Precuneus/Parietal Cortex", "Posterior Cingulate", "Anterior Cingulate",
            "Orbitofrontal Cortex", "Temporal Lobe", "Hippocampus/MTL",
            "Thalamus/Basal Ganglia"
        ]

        # Mixed/special regions
        if anatomical_region in white_matter_regions:
            return "White Matter"
        elif anatomical_region in gray_matter_regions:
            return "Gray Matter"
        elif anatomical_region == "Cerebellum":
            return "GM (Cerebellar)"
        elif anatomical_region == "Brainstem":
            return "Mixed (WM/GM)"
        else:
            return "Mixed"

    def analyze_intensity_profile(self, cam_norm, brain_data):
        """Analyze intensity profile in activated regions to distinguish GM vs WM."""
        if brain_data is None:
            return "N/A", "N/A"

        # Get high activation regions (top 20%)
        threshold = np.percentile(cam_norm, 80)
        high_activation_mask = cam_norm > threshold

        # Get brain intensities in those regions
        brain_norm = (brain_data - brain_data.min()) / (brain_data.max() - brain_data.min() + 1e-8)
        activated_intensities = brain_norm[high_activation_mask]

        if len(activated_intensities) == 0:
            return "N/A", "N/A"

        mean_intensity = activated_intensities.mean()
        std_intensity = activated_intensities.std()

        # In T1-weighted MRI:
        # Gray Matter: ~0.4-0.6 normalized intensity
        # White Matter: ~0.6-0.8 normalized intensity
        # CSF: ~0.1-0.3 normalized intensity

        if mean_intensity < 0.35:
            profile = "Low (CSF-like)"
        elif mean_intensity < 0.55:
            profile = "Medium (GM-like)"
        elif mean_intensity < 0.75:
            profile = "High (WM-like)"
        else:
            profile = "Very High"

        # Estimate GM/WM ratio based on intensity distribution
        gm_voxels = np.sum((activated_intensities >= 0.35) & (activated_intensities < 0.55))
        wm_voxels = np.sum((activated_intensities >= 0.55) & (activated_intensities < 0.80))

        if wm_voxels > 0:
            ratio = gm_voxels / wm_voxels
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "∞ (All GM)"

        return profile, ratio_str

    def validate_tissue_prediction(self, predicted_tissue, expected_tissue, anatomical_region, gm_wm_ratio):
        """Validate if the predicted tissue matches the anatomical expectations."""
        warnings = []

        # Check for anatomical contradictions
        if predicted_tissue == "Gray Matter":
            if "White Matter" in expected_tissue:
                warnings.append(f"⚠️ ANATOMICAL CONFLICT: Model predicts Gray Matter but focus is on {anatomical_region} "
                               f"which is a White Matter structure!")
            if "Corpus Callosum" in anatomical_region:
                warnings.append("🔴 CRITICAL: Corpus Callosum is the largest White Matter tract in the brain. "
                               "Gray Matter prediction here indicates potential model misalignment.")

        elif predicted_tissue == "White Matter":
            if "Gray Matter" in expected_tissue or expected_tissue == "GM (Cerebellar)":
                warnings.append(f"⚠️ ANATOMICAL CONFLICT: Model predicts White Matter but focus is on {anatomical_region} "
                               f"which is a Gray Matter structure!")

        # Check intensity profile consistency
        try:
            if gm_wm_ratio != "N/A" and gm_wm_ratio != "∞ (All GM)":
                ratio = float(gm_wm_ratio)
                if predicted_tissue == "Gray Matter" and ratio < 0.5:
                    warnings.append(f"⚠️ INTENSITY WARNING: Low GM/WM ratio ({ratio:.2f}) suggests focus region "
                                   "contains more White Matter than Gray Matter voxels.")
                elif predicted_tissue == "White Matter" and ratio > 2.0:
                    warnings.append(f"⚠️ INTENSITY WARNING: High GM/WM ratio ({ratio:.2f}) suggests focus region "
                                   "contains more Gray Matter than White Matter voxels.")
        except (ValueError, TypeError):
            pass

        return warnings

    def display_gradcam_views(self):
        """Display Grad-CAM in Axial, Coronal, and Sagittal views."""
        if self.cam_heatmap is None or self.display_data is None:
            return

        data = self.display_data
        cam = self.cam_heatmap
        d, h, w = data.shape

        # Get slice position from slider (0-100%)
        slice_pct = self.gradcam_slice_var.get() / 100.0

        # Clear axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.axis('off')

        # Custom colormap with transparency built in
        from matplotlib.colors import LinearSegmentedColormap
        colors_hot = ['#000033', '#0000aa', '#00aaff', '#00ff88', '#ffff00', '#ff8800', '#ff0000']
        custom_cmap = LinearSegmentedColormap.from_list('brain_heat', colors_hot, N=256)

        # Helper function to create masked RGBA heatmap
        def create_heatmap_rgba(cam_slice, mask, cmap):
            """Create RGBA heatmap with mask applied to alpha channel."""
            cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-8)
            rgba = cmap(cam_norm)
            rgba[..., 3] = mask * 0.7
            return rgba

        # Sagittal view (side view, X slice) - ax1 (LEFT)
        slice_idx = int(slice_pct * (w - 1))
        brain_raw = np.rot90(data[:, :, slice_idx])
        brain_slice = self.enhance_brain_slice(brain_raw)
        cam_slice = np.rot90(cam[:, :, slice_idx])
        brain_mask = self.create_brain_mask(brain_raw)
        heatmap_rgba = create_heatmap_rgba(cam_slice, brain_mask, custom_cmap)
        self.ax1.imshow(brain_slice, cmap='bone', aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        self.ax1.imshow(heatmap_rgba, aspect='equal', interpolation='bilinear')
        self.ax1.set_title(f'Sagittal (X={slice_idx})', color='#00d4ff', fontsize=11, fontweight='bold', pad=5)
        self.ax1.axis('off')

        # Coronal view (front view, Y slice) - ax2 (CENTER)
        slice_idx = int(slice_pct * (h - 1))
        brain_raw = np.rot90(data[:, slice_idx, :])
        brain_slice = self.enhance_brain_slice(brain_raw)
        cam_slice = np.rot90(cam[:, slice_idx, :])
        brain_mask = self.create_brain_mask(brain_raw)
        heatmap_rgba = create_heatmap_rgba(cam_slice, brain_mask, custom_cmap)
        self.ax2.imshow(brain_slice, cmap='bone', aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        self.ax2.imshow(heatmap_rgba, aspect='equal', interpolation='bilinear')
        self.ax2.set_title(f'Coronal (Y={slice_idx})', color='#00d4ff', fontsize=11, fontweight='bold', pad=5)
        self.ax2.axis('off')

        # Axial view (top-down, Z slice) - ax3 (RIGHT)
        slice_idx = int(slice_pct * (d - 1))
        brain_raw = np.rot90(data[slice_idx, :, :])
        brain_slice = self.enhance_brain_slice(brain_raw)
        cam_slice = np.rot90(cam[slice_idx, :, :])
        brain_mask = self.create_brain_mask(brain_raw)
        heatmap_rgba = create_heatmap_rgba(cam_slice, brain_mask, custom_cmap)
        self.ax3.imshow(brain_slice, cmap='bone', aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        self.ax3.imshow(heatmap_rgba, aspect='equal', interpolation='bilinear')
        self.ax3.set_title(f'Axial (Z={slice_idx})', color='#00d4ff', fontsize=11, fontweight='bold', pad=5)
        self.ax3.axis('off')

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

        # Update statistics with anatomical analysis
        stats = self.compute_gradcam_stats(cam, brain_data=data)
        for key, value in stats.items():
            if key in self.stat_labels:
                self.stat_labels[key].configure(text=value)

        # Get current tissue prediction and validate
        predicted_tissue = self.tissue_value.cget("text") if hasattr(self, 'tissue_value') else ""
        expected_tissue = stats.get('expected_tissue', '')
        anatomical_region = stats.get('anatomical_region', '')
        gm_wm_ratio = stats.get('gm_wm_ratio', 'N/A')

        # Run validation
        warnings = self.validate_tissue_prediction(predicted_tissue, expected_tissue,
                                                   anatomical_region, gm_wm_ratio)

        # Update validation label
        if warnings:
            warning_text = "\n".join(warnings)
            self.validation_label.configure(text=warning_text, fg='#FF6B6B')
        else:
            if predicted_tissue and expected_tissue:
                self.validation_label.configure(
                    text=f"✅ VALIDATION PASSED: {predicted_tissue} prediction is consistent with "
                         f"{anatomical_region} ({expected_tissue} structure)",
                    fg='#4CAF50'
                )
            else:
                self.validation_label.configure(text="", fg='#FFD93D')


def main():
    root = tk.Tk()
    app = BrainMRIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
