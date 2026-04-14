"""
Results Visualization Script for Conference Paper
=================================================
Generates a high-resolution figure showing model performance on sample images.

Layout: 3 rows (samples) x 4 columns:
    Column A: Original RGB Image
    Column B: Ground Truth Heatmap
    Column C: Predicted Heatmap (graspability glow)
    Column D: Final Predicted Grasp (green box on image)

Output: paper_results.png (300 DPI) and paper_results.pdf
"""

import os
import glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import GraspHeatmapNet
from dataset import GraspDataset

# Fixed gripper jaw height (pixels at 320x320 scale)
GRIPPER_HEIGHT_PX = 20


def load_model(model_path, device):
    """Loads the trained DeepLabV3+ model."""
    model = GraspHeatmapNet(pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using random weights.")
    model = model.to(device)
    model.eval()
    return model


def draw_grasp_on_image(rgb_img, heatmap, sin_map, cos_map, width_map, map_size=320):
    """Draws the predicted grasp box on the RGB image. Returns annotated image."""
    display = cv2.resize(rgb_img, (map_size, map_size))
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    
    # Find peak in heatmap
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    py, px = peak_idx
    confidence = heatmap[py, px]
    
    # Read parameters at peak
    sin_2t = sin_map[py, px]
    cos_2t = cos_map[py, px]
    gripper_w = width_map[py, px] * map_size
    theta_deg = np.degrees(np.arctan2(sin_2t, cos_2t) / 2.0)
    
    # Draw rotated rectangle
    rect = ((float(px), float(py)), (float(max(gripper_w, 5)), float(GRIPPER_HEIGHT_PX)), float(theta_deg))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(display, [box], 0, (0, 255, 0), 2)
    
    # Draw center marker
    cv2.circle(display, (int(px), int(py)), 4, (255, 0, 0), -1)
    
    # Confidence text
    cv2.putText(display, f"Conf: {confidence:.2f}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return display


def create_paper_figure(data_dir, model_path, output_dir='.', sample_indices=None):
    """
    Creates the 3x4 results figure for the conference paper.
    
    Args:
        data_dir:       Path to processed_data folder
        model_path:     Path to best_grasp_heatmap.pth
        output_dir:     Where to save the output figure
        sample_indices: List of 3 dataset indices to visualize. If None, picks automatically.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load dataset (full, no splitting)
    dataset = GraspDataset(data_dir)
    print(f"Dataset size: {len(dataset)} images")
    
    # Pick 3 sample indices (spread across dataset for variety)
    if sample_indices is None:
        n = len(dataset)
        sample_indices = [0, n // 3, 2 * n // 3]
    
    # Collect data for the figure
    rows_data = []
    
    for idx in sample_indices:
        img_tensor, gt_target = dataset[idx]
        
        # Get path for raw RGB
        img_path = dataset.image_files[idx]
        raw_rgb = cv2.imread(img_path)
        
        # Run inference
        img_input = img_tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = model(img_input)
        
        pred = output.squeeze(0).cpu().numpy()   # (4, 320, 320)
        gt   = gt_target.numpy()                  # (4, 320, 320)
        
        # A) Original RGB (resized to 320x320, displayed as RGB)
        rgb_display = cv2.resize(raw_rgb, (320, 320))
        rgb_display = cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB)
        
        # B) Ground Truth Heatmap
        gt_heatmap = gt[0]
        
        # C) Predicted Heatmap
        pred_heatmap = pred[0]
        
        # D) Predicted Grasp Box on image
        grasp_img = draw_grasp_on_image(
            raw_rgb, pred[0], pred[1], pred[2], pred[3]
        )
        
        rows_data.append({
            'rgb': rgb_display,
            'gt_heatmap': gt_heatmap,
            'pred_heatmap': pred_heatmap,
            'grasp_img': grasp_img,
            'idx': idx,
        })
    
    # === Create the matplotlib figure ===
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Column titles
    col_titles = [
        '(A) Original RGB Image',
        '(B) Ground Truth Heatmap',
        '(C) Predicted Heatmap',
        '(D) Predicted Grasp',
    ]
    
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    for row, data in enumerate(rows_data):
        # Label rows
        axes[row, 0].set_ylabel(f"Sample {data['idx']}", fontsize=12, fontweight='bold',
                                 rotation=90, labelpad=15)
        
        # Column A: RGB
        axes[row, 0].imshow(data['rgb'])
        axes[row, 0].axis('off')
        
        # Column B: GT Heatmap (viridis colormap for paper quality)
        im_gt = axes[row, 1].imshow(data['gt_heatmap'], cmap='inferno', vmin=0, vmax=1)
        axes[row, 1].axis('off')
        
        # Column C: Predicted Heatmap
        im_pred = axes[row, 2].imshow(data['pred_heatmap'], cmap='inferno', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        
        # Column D: Grasp visualization
        axes[row, 3].imshow(data['grasp_img'])
        axes[row, 3].axis('off')
    
    # Add a single colorbar for heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im_pred, cax=cbar_ax)
    cbar.set_label('Graspability Score', fontsize=12)
    
    # Overall title
    fig.suptitle(
        'DeepLabV3+ with ASPP: Heatmap-Based Grasp Detection Results\n'
        'ResNet-50 Encoder | ASPP Rates [6, 12, 18] | ivalab/grasp_multiObject Dataset',
        fontsize=15, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    
    # Save as high-resolution PNG and PDF
    png_path = os.path.join(output_dir, 'paper_results.png')
    pdf_path = os.path.join(output_dir, 'paper_results.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nPaper figure saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    plt.close(fig)
    
    # === Also create a training metrics plot if history exists ===
    history_path = os.path.join(output_dir, 'training_history.pth')
    if os.path.exists(history_path):
        create_metrics_plot(history_path, output_dir)


def create_metrics_plot(history_path, output_dir='.'):
    """
    Creates a 2x2 training metrics plot for the paper:
        Top-left:     Loss curves (train/val)
        Top-right:    Mean IoU over epochs
        Bottom-left:  Success Rate over epochs
        Bottom-right: F1 / Precision / Recall over epochs
    """
    history = torch.load(history_path, weights_only=False)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for consistency
    c_train = '#2196F3'
    c_val   = '#F44336'
    c_miou  = '#4CAF50'
    c_sr    = '#FF9800'
    c_f1    = '#9C27B0'
    c_prec  = '#00BCD4'
    c_rec   = '#E91E63'
    
    # Top-left: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=c_train, linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'], color=c_val, linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Mean IoU
    ax = axes[0, 1]
    ax.plot(epochs, history['miou'], color=c_miou, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('mIoU', fontsize=11)
    ax.set_title('Mean IoU (Heatmap Channel)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Success Rate
    ax = axes[1, 0]
    ax.plot(epochs, history['success_rate'], color=c_sr, linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title('Grasp Success Rate (10px, 15 deg)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: F1 / Precision / Recall
    ax = axes[1, 1]
    ax.plot(epochs, history['f1'], color=c_f1, linewidth=2, label='F1-Score')
    ax.plot(epochs, history['precision'], color=c_prec, linewidth=1.5, linestyle='--', label='Precision')
    ax.plot(epochs, history['recall'], color=c_rec, linewidth=1.5, linestyle='--', label='Recall')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Detection F1 / Precision / Recall', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        'DeepLabV3+ Grasp Detection: Training Metrics',
        fontsize=15, fontweight='bold', y=1.01
    )
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'paper_metrics.png')
    pdf_path = os.path.join(output_dir, 'paper_metrics.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nMetrics plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    DATA_DIR   = r"d:\Grasp_-CNN\processed_data"
    MODEL_PATH = r"d:\Grasp_-CNN\best_grasp_heatmap.pth"
    OUTPUT_DIR = r"d:\Grasp_-CNN"
    
    create_paper_figure(
        data_dir=DATA_DIR,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        sample_indices=[0, 32, 64]  # Spread across dataset for variety
    )
