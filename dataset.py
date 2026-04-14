import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2

# Gaussian radius in pixels for ground-truth heatmap blobs
GAUSSIAN_SIGMA = 5

class GraspDataset(Dataset):
    def __init__(self, data_dir, map_size=320):
        """
        Heatmap-based grasp dataset.
        
        For each image, generates 4-channel ground-truth maps (map_size x map_size):
            Ch 0: Graspability heatmap — 2D Gaussian at each grasp center (cx, cy)
            Ch 1: sin(2*theta) map    — filled at grasp regions
            Ch 2: cos(2*theta) map    — filled at grasp regions
            Ch 3: Normalized width    — w / map_size, filled at grasp regions
            
        Args:
            data_dir (str): Directory containing 'images' and 'labels' subfolders.
            map_size (int): Spatial resolution of the output maps (default 320).
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'images')
        self.lbl_dir = os.path.join(data_dir, 'labels')
        self.map_size = map_size
        
        # Assuming image and label filenames match, e.g., rgd_0000.png <-> rgd_0000.txt
        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, '*.png')) + 
                                  glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
    def __len__(self):
        return len(self.image_files)
    
    def _generate_gaussian(self, cx, cy, sigma=GAUSSIAN_SIGMA):
        """
        Generates a 2D Gaussian blob centered at (cx, cy) on a (map_size x map_size) grid.
        Returns a numpy array of shape (map_size, map_size) with values in [0, 1].
        """
        size = self.map_size
        x = np.arange(0, size, dtype=np.float32)
        y = np.arange(0, size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)  # yy is rows, xx is cols
        
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        return gaussian
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, f"{base_name}.txt")
        
        # Load the RG-D image using OpenCV
        image = cv2.imread(img_path)
        # Transform BGR back to the standard RGB logic (Red, Green, Depth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert axes for Pytorch [C, H, W] and normalize pixel values to [0, 1]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        
        # Initialize the 4 ground-truth maps
        heatmap    = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        sin_map    = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        cos_map    = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        width_map  = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        # Parse the labels and paint onto the maps
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx = float(parts[0])
                        cy = float(parts[1])
                        w  = float(parts[2])
                        # h  = float(parts[3])  # Not used — fixed gripper height
                        theta = float(parts[4])
                        
                        # Clamp center coordinates to valid pixel range
                        cx = np.clip(cx, 0, self.map_size - 1)
                        cy = np.clip(cy, 0, self.map_size - 1)
                        
                        # Generate Gaussian blob for this grasp
                        gaussian = self._generate_gaussian(cx, cy)
                        
                        # Take element-wise max so overlapping grasps don't cancel out
                        heatmap = np.maximum(heatmap, gaussian)
                        
                        # Compute angle encoding
                        sin_2theta = np.sin(2 * theta)
                        cos_2theta = np.cos(2 * theta)
                        
                        # Fill regression maps within the Gaussian's effective radius
                        # Use a threshold to define the "active" region for this grasp
                        mask = gaussian > 0.01
                        sin_map[mask]   = sin_2theta
                        cos_map[mask]   = cos_2theta
                        width_map[mask] = w / self.map_size  # Normalize width to [0, 1]
        
        # Stack into (4, H, W) target tensor
        target = np.stack([heatmap, sin_map, cos_map, width_map], axis=0)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def get_dataloaders(data_dir, batch_size=4, val_split=0.2, num_workers=0):
    """
    Creates train and validation dataloaders for heatmap-based grasp detection.
    All targets are now fixed-shape tensors (4, 320, 320), so default collation works.
    """
    dataset = GraspDataset(data_dir=data_dir)
    
    # Calculate sizes for the 20% validation split
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Check if dataset is reasonably sized to avoid crash
    if dataset_size == 0:
        raise ValueError(f"No images found in {data_dir}/images.")
        
    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Reproducibility seed
    )
    
    # Train DataLoader — no custom collate_fn needed anymore
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Critical for quickly moving data onto your RTX 4060
    )
    
    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    data_directory = r"d:\Grasp_-CNN\processed_data"
    
    # A lightweight test if the directory actually exists
    if os.path.exists(os.path.join(data_directory, 'images')):
        train_ld, val_ld = get_dataloaders(data_directory, batch_size=4)
        print(f"Number of training batches: {len(train_ld)}")
        print(f"Number of validation batches: {len(val_ld)}")
        
        for images, targets in train_ld:
            print(f"Image tensor shape: {images.shape}")      # Expected: (4, 3, 320, 320)
            print(f"Target maps shape:  {targets.shape}")      # Expected: (4, 4, 320, 320)
            print(f"Heatmap max value:  {targets[0, 0].max():.4f}")
            print(f"Heatmap min value:  {targets[0, 0].min():.4f}")
            break
    else:
        print(f"Data directory {data_directory} not found. Ensure process_grasps.py finished running.")
