"""
Brain MRI Prediction GUI with Grad-CAM Explainable AI
=====================================================
Interactive GUI to predict Age, Sex, and Tissue Type from brain MRI scans.
Uses 3D CNN model with Grad-CAM to highlight important brain regions.

Run: python brain_mri_gui.py
"""

import os
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
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install with: pip install nibabel pillow torch")
    exit(1)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainCNN3D(nn.Module):
    """3D CNN for multi-task brain MRI prediction."""

    def __init__(self):
        super(BrainCNN3D, self).__init__()

        # Feature extractor (3D CNN)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 64 -> 32

            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32 -> 16

            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16 -> 8

            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
        )

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Task-specific heads
        self.age_head = nn.Linear(128, 1)      # Age regression
        self.sex_head = nn.Linear(128, 1)      # Sex classification
        self.tissue_head = nn.Linear(128, 1)   # Tissue classification

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        age = self.age_head(x)
        sex = torch.sigmoid(self.sex_head(x))
        tissue = torch.sigmoid(self.tissue_head(x))

        return age, sex, tissue


class GradCAM3D:
    """Grad-CAM for 3D CNN models."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target='age'):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        input_tensor.requires_grad = True

        # Forward pass
        age, sex, tissue = self.model(input_tensor)

        # Select target output
        if target == 'age':
            output = age
        elif target == 'sex':
            output = sex
        else:
            output = tissue

        # Backward pass
        self.model.zero_grad()
        output.backward(retain_graph=True)

        # Get weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)

        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


class BrainMRIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain MRI Analyzer - Grad-CAM")
        self.root.geometry("1500x850")
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
        self.model.eval()

        # Setup Grad-CAM on the last conv layer (Block 4)
        target_layer = self.model.features[12]  # Block 4 Conv3d
        self.grad_cam = GradCAM3D(self.model, target_layer)
        self.current_model_type = "cnn"

        print(f"CNN Model loaded successfully!")
        print(f"  Age MAE: {checkpoint.get('age_mae', 'N/A'):.2f} years")

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
        """Create all GUI widgets."""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a2e')
        title_frame.pack(pady=20)

        title = tk.Label(title_frame, text="🧠 Brain MRI Analyzer",
                        font=('Segoe UI', 28, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        title.pack()

        subtitle = tk.Label(title_frame, text="Predict Age, Sex & Tissue Type from MRI Scans",
                           font=('Segoe UI', 12), fg='#888888', bg='#1a1a2e')
        subtitle.pack()

        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Original Image + Results
        left_panel = tk.Frame(main_frame, bg='#16213e', relief=tk.FLAT, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        # Original Image Title
        img_title = tk.Label(left_panel, text="🧠 Original MRI Scan",
                            font=('Segoe UI', 14, 'bold'), fg='#00d4ff', bg='#16213e')
        img_title.pack(pady=(15, 10))

        # Image display frame with border
        img_frame = tk.Frame(left_panel, bg='#0f3460', padx=3, pady=3)
        img_frame.pack(padx=15)

        self.image_label = tk.Label(img_frame, text="No Image Loaded\n\nClick 'Load MRI' to start",
                                    font=('Segoe UI', 11), fg='#666666', bg='#1a1a2e')
        self.image_label.pack()

        # Slice slider
        slider_frame = tk.Frame(left_panel, bg='#16213e')
        slider_frame.pack(fill=tk.X, padx=20, pady=(10, 5))

        self.slice_var = tk.IntVar(value=50)
        self.slice_slider = ttk.Scale(slider_frame, from_=0, to=100, variable=self.slice_var,
                                      orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.pack(fill=tk.X)

        self.slice_label = tk.Label(left_panel, text="Slice: 50", font=('Segoe UI', 10),
                                    fg='#00d4ff', bg='#16213e')
        self.slice_label.pack(pady=(0, 10))

        # Separator
        sep = tk.Frame(left_panel, height=2, bg='#0f3460')
        sep.pack(fill=tk.X, padx=20, pady=10)

        # Result cards in left panel
        results_title = tk.Label(left_panel, text="📊 Prediction Results",
                                font=('Segoe UI', 14, 'bold'), fg='#00d4ff', bg='#16213e')
        results_title.pack(pady=(5, 10))

        self.create_result_card(left_panel, "age")
        self.create_result_card(left_panel, "sex")
        self.create_result_card(left_panel, "tissue")

        # Right panel - Grad-CAM Visualization
        right_panel = tk.Frame(main_frame, bg='#16213e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

        explain_title = tk.Label(right_panel, text="🔬 Grad-CAM: Brain Region Importance",
                                font=('Segoe UI', 18, 'bold'), fg='#00d4ff', bg='#16213e')
        explain_title.pack(pady=(15, 5))

        explain_sub = tk.Label(right_panel, text="Highlighting which brain regions influenced the prediction",
                              font=('Segoe UI', 10), fg='#888888', bg='#16213e')
        explain_sub.pack(pady=(0, 10))

        # Target selector with styled buttons
        target_frame = tk.Frame(right_panel, bg='#16213e')
        target_frame.pack(pady=10)

        tk.Label(target_frame, text="Explain prediction for:", font=('Segoe UI', 11, 'bold'),
                fg='#ffffff', bg='#16213e').pack(side=tk.LEFT, padx=(0, 15))

        self.target_var = tk.StringVar(value='age')
        for text, val, emoji in [("👤 Age", "age", ""), ("⚤ Sex", "sex", ""), ("🧬 Tissue", "tissue", "")]:
            rb = tk.Radiobutton(target_frame, text=text, variable=self.target_var, value=val,
                               bg='#16213e', fg='white', selectcolor='#0f3460',
                               font=('Segoe UI', 11), indicatoron=0, padx=15, pady=5,
                               activebackground='#00d4ff', activeforeground='#1a1a2e',
                               command=self.update_gradcam)
            rb.pack(side=tk.LEFT, padx=5)

        # Matplotlib figure for Grad-CAM - larger size with high DPI
        self.fig = Figure(figsize=(12, 7), facecolor='#16213e', dpi=100)
        self.fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.05)

        # Three views: Axial, Coronal, Sagittal
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

        for ax, title in [(self.ax1, 'Axial (Top View)'), (self.ax2, 'Coronal (Front View)'), (self.ax3, 'Sagittal (Side View)')]:
            ax.set_facecolor('#16213e')
            ax.set_title(title, color='#00d4ff', fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Colorbar info with better styling
        info_frame = tk.Frame(right_panel, bg='#0f3460', padx=15, pady=8)
        info_frame.pack(pady=(0, 10))
        info_label = tk.Label(info_frame, text="🔴 Red/Yellow = High Importance    |    🔵 Blue/Green = Low Importance",
                             font=('Segoe UI', 11), fg='white', bg='#0f3460')
        info_label.pack()

        # Buttons at bottom
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=10)

        # Model selector dropdown
        model_frame = tk.Frame(btn_frame, bg='#1a1a2e')
        model_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(model_frame, text="Model:", font=('Segoe UI', 10, 'bold'),
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
                                         values=model_options, state='readonly', width=25)
            self.model_dropdown.pack(side=tk.LEFT)
            # Use trace for more reliable callback
            self.model_var.trace_add('write', self.switch_model)

        # Grad-CAM status indicator
        status_text = "✅ Grad-CAM Available" if self.cnn_available else "❌ Grad-CAM Not Available"
        status_color = '#4CAF50' if self.cnn_available else '#FF6B6B'
        self.gradcam_status = tk.Label(btn_frame, text=status_text, font=('Segoe UI', 9),
                                       fg=status_color, bg='#1a1a2e')
        self.gradcam_status.pack(side=tk.LEFT, padx=10)

        load_btn = tk.Button(btn_frame, text="📁 Load MRI Image", font=('Segoe UI', 12, 'bold'),
                            bg='#00d4ff', fg='#1a1a2e', padx=20, pady=10,
                            cursor='hand2', command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=10)

        predict_btn = tk.Button(btn_frame, text="🔍 Predict & Explain", font=('Segoe UI', 12, 'bold'),
                               bg='#4CAF50', fg='white', padx=20, pady=10,
                               cursor='hand2', command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=10)

    def create_result_card(self, parent, result_type):
        """Create a result display card."""
        colors = {
            'age': ('#FF6B6B', '👤'),
            'sex': ('#4ECDC4', '⚤'),
            'tissue': ('#95E1D3', '🧬')
        }
        color, icon = colors[result_type]

        card = tk.Frame(parent, bg='#0f3460', relief=tk.FLAT)
        card.pack(fill=tk.X, padx=15, pady=8)

        # Icon and label
        header = tk.Frame(card, bg='#0f3460')
        header.pack(fill=tk.X, padx=10, pady=(10, 5))

        icon_label = tk.Label(header, text=icon, font=('Segoe UI', 20), bg='#0f3460', fg=color)
        icon_label.pack(side=tk.LEFT)

        labels = {'age': 'Age', 'sex': 'Sex', 'tissue': 'Tissue Type'}
        name_label = tk.Label(header, text=labels[result_type], font=('Segoe UI', 12),
                             bg='#0f3460', fg='#cccccc')
        name_label.pack(side=tk.LEFT, padx=10)

        # Value
        value_label = tk.Label(card, text="---", font=('Segoe UI', 24, 'bold'),
                              bg='#0f3460', fg=color)
        value_label.pack(pady=(0, 10))

        # Store reference
        setattr(self, f'{result_type}_value', value_label)

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
                self.image_label.configure(text="")
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

        # Apply colormap for better visualization (bone colormap effect)
        img = Image.fromarray(slice_data, mode='L')
        img = img.resize((320, 320), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo, text="")

    def preprocess_for_cnn(self):
        """Preprocess image for CNN model."""
        data = self.current_image_data.astype(np.float32)

        # Resize to target size for model
        factors = [t / s for t, s in zip(self.target_size, data.shape)]
        data = zoom(data, factors, order=1)

        # Normalize to 0-1
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # Store resized data for model
        self.resized_data = data

        # Create higher resolution version for display
        display_data = self.current_image_data.astype(np.float32)
        display_factors = [t / s for t, s in zip(self.display_size, display_data.shape)]
        display_data = zoom(display_data, display_factors, order=3)  # cubic interpolation
        display_data = (display_data - display_data.min()) / (display_data.max() - display_data.min() + 1e-8)
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

            # Update UI
            self.age_value.configure(text=f"{int(round(age_pred[0]))} years")
            self.sex_value.configure(text="Female" if sex_pred[0] == 'F' else "Male")
            self.tissue_value.configure(text="Gray Matter" if tissue_pred[0] == 'GM' else "White Matter")

            # Clear Grad-CAM (not available for sklearn)
            self.cam_heatmap = None
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.clear()
                ax.set_facecolor('#16213e')
                ax.axis('off')
            self.ax2.text(0.5, 0.5, "Grad-CAM not available\nfor Sklearn model\n\nSwitch to CNN model\nfor visualizations",
                         ha='center', va='center', fontsize=14, color='#888888',
                         transform=self.ax2.transAxes)
            self.canvas.draw()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Sklearn prediction failed:\n{e}")

    def predict_cnn(self):
        """Run prediction using CNN model with Grad-CAM."""
        if self.model is None:
            messagebox.showerror("Error", "CNN Model not loaded!\nRun train_brain_cnn.py first.")
            return

        try:
            # Preprocess image
            self.current_tensor = self.preprocess_for_cnn()

            # Run prediction
            self.model.eval()
            with torch.no_grad():
                age_pred, sex_pred, tissue_pred = self.model(self.current_tensor)

            # Convert predictions
            age = int(round(age_pred.item() * 100))
            sex = 'F' if sex_pred.item() > 0.5 else 'M'
            tissue = 'WM' if tissue_pred.item() > 0.5 else 'GM'

            # Update UI
            self.age_value.configure(text=f"{age} years")
            self.sex_value.configure(text="Female" if sex == 'F' else "Male")
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

    def display_gradcam_views(self):
        """Display Grad-CAM in Axial, Coronal, and Sagittal views."""
        if self.cam_heatmap is None or self.display_data is None:
            return

        data = self.display_data
        cam = self.cam_heatmap
        d, h, w = data.shape

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
            # Normalize cam to 0-1
            cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-8)
            # Get RGBA from colormap
            rgba = cmap(cam_norm)
            # Apply mask to alpha channel - zero alpha outside brain
            rgba[..., 3] = mask * 0.7  # 0.7 max alpha for overlay
            return rgba

        # Axial view (top-down, Z slice)
        slice_idx = d // 2
        brain_raw = np.rot90(data[slice_idx, :, :])
        brain_slice = self.enhance_brain_slice(brain_raw)
        cam_slice = np.rot90(cam[slice_idx, :, :])
        brain_mask = self.create_brain_mask(brain_raw)
        heatmap_rgba = create_heatmap_rgba(cam_slice, brain_mask, custom_cmap)
        self.ax1.imshow(brain_slice, cmap='bone', aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        self.ax1.imshow(heatmap_rgba, aspect='equal', interpolation='bilinear')
        self.ax1.set_title('Axial', color='#00d4ff', fontsize=13, fontweight='bold', pad=8)
        self.ax1.axis('off')

        # Coronal view (front view, Y slice)
        slice_idx = h // 2
        brain_raw = np.rot90(data[:, slice_idx, :])
        brain_slice = self.enhance_brain_slice(brain_raw)
        cam_slice = np.rot90(cam[:, slice_idx, :])
        brain_mask = self.create_brain_mask(brain_raw)
        heatmap_rgba = create_heatmap_rgba(cam_slice, brain_mask, custom_cmap)
        self.ax2.imshow(brain_slice, cmap='bone', aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        self.ax2.imshow(heatmap_rgba, aspect='equal', interpolation='bilinear')
        self.ax2.set_title('Coronal', color='#00d4ff', fontsize=13, fontweight='bold', pad=8)
        self.ax2.axis('off')

        # Sagittal view (side view, X slice)
        slice_idx = w // 2
        brain_raw = np.rot90(data[:, :, slice_idx])
        brain_slice = self.enhance_brain_slice(brain_raw)
        cam_slice = np.rot90(cam[:, :, slice_idx])
        brain_mask = self.create_brain_mask(brain_raw)
        heatmap_rgba = create_heatmap_rgba(cam_slice, brain_mask, custom_cmap)
        self.ax3.imshow(brain_slice, cmap='bone', aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        self.ax3.imshow(heatmap_rgba, aspect='equal', interpolation='bilinear')
        self.ax3.set_title('Sagittal', color='#00d4ff', fontsize=13, fontweight='bold', pad=8)
        self.ax3.axis('off')

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = BrainMRIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
