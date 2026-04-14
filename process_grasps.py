import os
import glob
import cv2
import numpy as np

def points_to_grasp(points):
    """
    Converts 4 sequence corner points into [x, y, w, h, theta].
    points structure: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    
    # Center is the mean of all 4 corners
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    
    # Width is distance from p0 to p1
    w = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    # Height is distance from p1 to p2
    h = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # Theta is the angle of the edge p0->p1
    theta = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    
    return [cx, cy, w, h, theta]


def process_data(rgbd_dir, annotation_dir, output_dir):
    out_img_dir = os.path.join(output_dir, 'images')
    out_lbl_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    # Target size
    target_size = (320, 320)
    
    # Get all RGB images
    rgb_files = glob.glob(os.path.join(rgbd_dir, 'rgb_*.png'))
    if not rgb_files:
        rgb_files = glob.glob(os.path.join(rgbd_dir, 'rgb_*.jpg'))
        
    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        base_id = filename.split('_')[1].split('.')[0] # e.g. "0000"
        
        depth_path = os.path.join(rgbd_dir, f'depth_{base_id}.png')
        # UPDATED: Reading .txt instead of .mat
        txt_path = os.path.join(annotation_dir, f'rgb_{base_id}_annotations.txt')
        
        if not os.path.exists(depth_path):
            print(f"Warning: Depth image not found for {filename}")
            continue
            
        if not os.path.exists(txt_path):
            print(f"Warning: Annotation .txt file not found for {filename}")
            continue
            
        # 1. Read images
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if rgb_img is None or depth_img is None:
            print(f"Warning: Could not read images for ID {base_id}")
            continue
            
        B, G, R = cv2.split(rgb_img)
        orig_h, orig_w = R.shape
        
        # 2. Normalize depth map
        if depth_img.dtype == np.uint16 or depth_img.max() > 255:
            depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            depth_norm = depth_img.astype(np.uint8)
            
        # 3. Create RG-D format
        rgd_img = cv2.merge([depth_norm, G, R]) 
        
        # 4. Resize to 320x320
        rgd_resized = cv2.resize(rgd_img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 5. Parse .txt annotations
        scaled_grasps = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
            # The .txt format contains 4 lines (points) per grasp object
            num_grasps = len(lines) // 4
            for i in range(num_grasps):
                try:
                    p0 = list(map(float, lines[i*4].strip().split()))
                    p1 = list(map(float, lines[i*4 + 1].strip().split()))
                    p2 = list(map(float, lines[i*4 + 2].strip().split()))
                    p3 = list(map(float, lines[i*4 + 3].strip().split()))
                    
                    points = np.array([p0, p1, p2, p3])
                    cx, cy, w, h, theta = points_to_grasp(points)
                    
                    # 6. Scale annotations heavily back to 320x320 grid
                    scale_x = target_size[0] / orig_w
                    scale_y = target_size[1] / orig_h
                    
                    cx_scaled = cx * scale_x
                    cy_scaled = cy * scale_y
                    w_scaled = w * scale_x
                    h_scaled = h * scale_y
                    
                    scaled_grasps.append([cx_scaled, cy_scaled, w_scaled, h_scaled, theta])
                except Exception as e:
                    print(f"Error parsing grasp {i} in file {txt_path}: {e}")
                    continue
        
        if len(scaled_grasps) == 0:
            print(f"Warning: No valid grasp data found in {txt_path}")
            continue

        # 7. Save the processed images and labels
        out_img_path = os.path.join(out_img_dir, f'rgd_{base_id}.png')
        cv2.imwrite(out_img_path, rgd_resized)
        
        out_txt_path = os.path.join(out_lbl_dir, f'rgd_{base_id}.txt')
        with open(out_txt_path, 'w') as f:
            for g in scaled_grasps:
                f.write(f"{g[0]:.4f} {g[1]:.4f} {g[2]:.4f} {g[3]:.4f} {g[4]:.4f}\n")
                
    print("Processing complete. Files successfully deployed to:", output_dir)

if __name__ == "__main__":
    # Path mappings
    DIRECTORY_PATH = r"d:\Grasp_-CNN"
    
    RGBD_DIRECTORY = os.path.join(DIRECTORY_PATH, 'rgbd')
    ANNOTATIONS_DIRECTORY = os.path.join(DIRECTORY_PATH, 'grasp_multiObject-master', 'annotations') 
    OUTPUT_DIRECTORY = os.path.join(DIRECTORY_PATH, 'processed_data')
    
    process_data(RGBD_DIRECTORY, ANNOTATIONS_DIRECTORY, OUTPUT_DIRECTORY)
