import os
import glob
import cv2
import numpy as np
import torch
from model import GraspHeatmapNet

# Fixed gripper jaw height (pixels at 320x320 scale)
GRIPPER_HEIGHT_PX = 20


def load_model(model_path, device):
    """Loads the trained GraspHeatmapNet encoder-decoder model."""
    model = GraspHeatmapNet(pretrained=False)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded {model_path}.")
    else:
        print(f"Warning: {model_path} not found. Running inference with UNTRAINED random weights.")
        
    model = model.to(device)
    model.eval()
    return model


def process_image(rgb_path, depth_path, target_size=(320, 320)):
    """Replicates the RG-D preprocessing from our pipeline."""
    # 1. Read images
    rgb_img = cv2.imread(rgb_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Splitting into Channels
    B, G, R = cv2.split(rgb_img)
    
    # 2. Normalize depth
    if depth_img.dtype == np.uint16 or depth_img.max() > 255:
        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        depth_norm = depth_img.astype(np.uint8)
        
    # 3. Stack into Depth, Green, Red formatting
    rgd_img = cv2.merge([depth_norm, G, R]) 
    
    # 4. Resize to exactly match the target expected size
    rgd_resized = cv2.resize(rgd_img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 5. Correct RGB matching logic for Tensor conversions
    tensor_img = cv2.cvtColor(rgd_resized, cv2.COLOR_BGR2RGB)
    
    # 6. Dimensions and Normalization -> [C, H, W], [0, 1]
    tensor_img = np.transpose(tensor_img, (2, 0, 1)).astype(np.float32) / 255.0
    tensor = torch.tensor(tensor_img).unsqueeze(0)  # Add batch dimension -> (1, 3, 320, 320)
    
    return tensor, rgb_img


def draw_grasp_from_heatmap(rgb_img, output_tensor, map_size=320,
                             window_name="Grasp Result", save_name="inference_result.png"):
    """
    Extracts the best grasp from the 4-channel heatmap output:
        Ch 0: Graspability heatmap  → find peak (best grasp location)
        Ch 1: sin(2θ) at peak      → recover angle
        Ch 2: cos(2θ) at peak      → recover angle  
        Ch 3: Normalized width      → recover gripper width
    """
    # Remove batch dim: (1, 4, 320, 320) → (4, 320, 320)
    preds = output_tensor.squeeze(0).cpu().numpy()
    
    heatmap   = preds[0]   # (320, 320) — graspability
    sin_map   = preds[1]   # (320, 320) — sin(2θ)
    cos_map   = preds[2]   # (320, 320) — cos(2θ)
    width_map = preds[3]   # (320, 320) — normalized width
    
    # --- Peak Detection: find the pixel with the highest graspability ---
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    py, px = peak_idx  # row (y), col (x) in the 320x320 grid
    confidence = heatmap[py, px]
    
    # --- Read grasp parameters at the peak location ---
    sin_2theta = sin_map[py, px]
    cos_2theta = cos_map[py, px]
    gripper_w  = width_map[py, px] * map_size  # De-normalize width
    gripper_h  = GRIPPER_HEIGHT_PX              # Fixed gripper jaw height
    
    # Recover angle: atan2(sin2θ, cos2θ) / 2
    theta_rad = np.arctan2(sin_2theta, cos_2theta) / 2.0
    theta_deg = np.degrees(theta_rad)
    
    print(f"\n--- Heatmap Peak Detection ---")
    print(f"  Peak location:  ({px}, {py})")
    print(f"  Confidence:     {confidence:.4f}")
    print(f"  Gripper width:  {gripper_w:.1f} px")
    print(f"  Grasp angle:    {theta_deg:.1f} deg")
    
    # --- Visualization ---
    display_img = cv2.resize(rgb_img, (map_size, map_size), interpolation=cv2.INTER_LINEAR)
    
    # 1. Draw the grasp rectangle
    rect = ((float(px), float(py)), (float(gripper_w), float(gripper_h)), float(theta_deg))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
    
    # Draw center crosshair
    cv2.circle(display_img, (int(px), int(py)), 4, (0, 0, 255), -1)
    
    # Draw text overlay
    text = f"Grasp (Conf: {confidence:.2f}, Angle: {theta_deg:.1f} deg)"
    cv2.putText(display_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(save_name, display_img)
    print(f"  Grasp visualization saved to '{save_name}'")
    
    # 2. Save a heatmap overlay for debugging
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.resize(rgb_img, (map_size, map_size)), 0.5, 
        heatmap_color, 0.5, 
        0
    )
    # Mark the peak on the overlay too
    cv2.circle(overlay, (int(px), int(py)), 6, (255, 255, 255), 2)
    
    overlay_save = save_name.replace('.png', '_heatmap_overlay.png')
    cv2.imwrite(overlay_save, overlay)
    print(f"  Heatmap overlay saved to '{overlay_save}'")
    
    # Attempt to display
    try:
        cv2.imshow(window_name, display_img)
        cv2.imshow("Heatmap Overlay", overlay)
        print("Displaying windows. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Note: GUI window not available, but images saved to disk.")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initialization: Attaching pipeline to {device}")
    
    rgbd_dir = r"d:\Grasp_-CNN\rgbd"
    model_path = r"d:\Grasp_-CNN\best_grasp_heatmap.pth"
    
    # 1. Grab sample image
    rgb_files = glob.glob(os.path.join(rgbd_dir, 'rgb_*.png'))
    if not rgb_files:
        rgb_files = glob.glob(os.path.join(rgbd_dir, 'rgb_*.jpg'))
        
    if not rgb_files:
        print(f"Error: Could not locate RGB files inside target folder: {rgbd_dir}")
        return
        
    sample_rgb = rgb_files[0]
    base_id = os.path.basename(sample_rgb).split('_')[1].split('.')[0]
    sample_depth = os.path.join(rgbd_dir, f'depth_{base_id}.png')
    
    print(f"Targeting Image Sample: {base_id}")
    
    # 2. Preprocess into tensor
    tensor_input, raw_rgb = process_image(sample_rgb, sample_depth)
    tensor_input = tensor_input.to(device)
    
    # 3. Load model
    model = load_model(model_path, device)
    
    # 4. Predict with mixed precision
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            output = model(tensor_input)  # (1, 4, 320, 320)
            
    # 5. Visualize grasp from heatmap peak
    draw_grasp_from_heatmap(raw_rgb, output)

if __name__ == "__main__":
    main()
