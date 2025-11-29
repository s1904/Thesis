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
        self.grad_cam = None
        self.current_image_data = None
        self.current_tensor = None
        self.cam_heatmap = None
        self.resized_data = None
        self.display_data = None  # Higher resolution for display
        self.current_filepath = None
        self.target_size = (64, 64, 64)
        self.display_size = (256, 256, 256)  # High resolution for visualization
        self.load_model()

        # Create UI
        self.create_widgets()

    def load_model(self):
        """Load the trained CNN model."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "brain_cnn_model.pth")

        if os.path.exists(model_path):
            self.model = BrainCNN3D().to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Setup Grad-CAM on the last conv layer (Block 4)
            # Target the last Conv3d layer before pooling
            target_layer = self.model.features[12]  # Block 4 Conv3d
            self.grad_cam = GradCAM3D(self.model, target_layer)

            print(f"CNN Model loaded successfully!")
            print(f"  Age MAE: {checkpoint.get('age_mae', 'N/A'):.2f} years")
        else:
            messagebox.showwarning("Warning", f"CNN model not found at:\n{model_path}\n\nRun train_brain_cnn.py first.")

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

    def display_gradcam_views(self):
        """Display Grad-CAM in Axial, Coronal, and Sagittal views."""
        if self.cam_heatmap is None or self.display_data is None:
            return

        data = self.display_data  # Use higher resolution data
        cam = self.cam_heatmap

        # Get center slices
        d, h, w = data.shape

        # Clear axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.axis('off')

        # Axial view (top-down, Z slice)
        slice_idx = d // 2
        brain_slice = np.rot90(data[slice_idx, :, :])
        cam_slice = np.rot90(cam[slice_idx, :, :])
        self.ax1.imshow(brain_slice, cmap='gray', aspect='equal', interpolation='lanczos')
        self.ax1.imshow(cam_slice, cmap='jet', alpha=0.45, aspect='equal', interpolation='gaussian')
        self.ax1.set_title('Axial', color='#00d4ff', fontsize=12, fontweight='bold')
        self.ax1.axis('off')

        # Coronal view (front view, Y slice)
        slice_idx = h // 2
        brain_slice = np.rot90(data[:, slice_idx, :])
        cam_slice = np.rot90(cam[:, slice_idx, :])
        self.ax2.imshow(brain_slice, cmap='gray', aspect='equal', interpolation='lanczos')
        self.ax2.imshow(cam_slice, cmap='jet', alpha=0.45, aspect='equal', interpolation='gaussian')
        self.ax2.set_title('Coronal', color='#00d4ff', fontsize=12, fontweight='bold')
        self.ax2.axis('off')

        # Sagittal view (side view, X slice)
        slice_idx = w // 2
        brain_slice = np.rot90(data[:, :, slice_idx])
        cam_slice = np.rot90(cam[:, :, slice_idx])
        self.ax3.imshow(brain_slice, cmap='gray', aspect='equal', interpolation='lanczos')
        self.ax3.imshow(cam_slice, cmap='jet', alpha=0.45, aspect='equal', interpolation='gaussian')
        self.ax3.set_title('Sagittal', color='#00d4ff', fontsize=12, fontweight='bold')
        self.ax3.axis('off')

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = BrainMRIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
