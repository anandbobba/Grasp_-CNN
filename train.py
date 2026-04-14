import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp import autocast, GradScaler
from dataset import get_dataloaders, GAUSSIAN_SIGMA
from model import GraspHeatmapNet


def masked_smooth_l1(pred, target, mask):
    """
    Computes SmoothL1Loss only at pixels where the mask is True.
    This restricts regression losses (angle, width) to actual grasp regions,
    preventing the model from wasting capacity on background pixels.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.functional.smooth_l1_loss(pred[mask], target[mask])


# ==============================================================================
# Scientific Metrics for Conference Paper
# ==============================================================================

def compute_mean_iou(pred_heatmap, gt_heatmap, threshold=0.5):
    """
    Computes Mean Intersection-over-Union (mIoU) on the heatmap channel.
    
    Binarizes both prediction and ground truth at the given threshold,
    then computes IoU = intersection / union for each image in the batch.
    Returns the mean IoU across the batch.
    """
    pred_binary = (pred_heatmap > threshold).float()
    gt_binary   = (gt_heatmap > threshold).float()
    
    # Per-image IoU
    ious = []
    for i in range(pred_binary.shape[0]):
        intersection = (pred_binary[i] * gt_binary[i]).sum()
        union = ((pred_binary[i] + gt_binary[i]) > 0).float().sum()
        
        if union == 0:
            # No grasp regions in GT or prediction — skip (perfect if both empty)
            ious.append(1.0 if intersection == 0 else 0.0)
        else:
            ious.append((intersection / union).item())
    
    return np.mean(ious) if ious else 0.0


def extract_grasp_centers(heatmap_2d, min_distance=10, threshold=0.3):
    """
    Extracts grasp center coordinates from a 2D heatmap using non-maximum suppression.
    Returns a list of (y, x) tuples for each detected peak.
    """
    # Simple NMS via max pooling
    heatmap_np = heatmap_2d.cpu().numpy() if isinstance(heatmap_2d, torch.Tensor) else heatmap_2d
    
    centers = []
    hm = heatmap_np.copy()
    
    while True:
        peak_val = hm.max()
        if peak_val < threshold:
            break
        
        py, px = np.unravel_index(np.argmax(hm), hm.shape)
        centers.append((py, px))
        
        # Suppress the region around this peak
        y_min = max(0, py - min_distance)
        y_max = min(hm.shape[0], py + min_distance + 1)
        x_min = max(0, px - min_distance)
        x_max = min(hm.shape[1], px + min_distance + 1)
        hm[y_min:y_max, x_min:x_max] = 0
    
    return centers


def compute_success_rate(pred_heatmap, pred_sin, pred_cos,
                         gt_heatmap, gt_sin, gt_cos,
                         center_thresh=10, angle_thresh=15.0):
    """
    Success Rate (SR): Percentage of images where the predicted grasp center
    is within `center_thresh` pixels of a true center AND the angle error
    is less than `angle_thresh` degrees.
    """
    batch_size = pred_heatmap.shape[0]
    successes = 0
    
    for i in range(batch_size):
        # Extract predicted peak
        pred_peaks = extract_grasp_centers(pred_heatmap[i], threshold=0.2)
        gt_peaks   = extract_grasp_centers(gt_heatmap[i], threshold=0.3)
        
        if len(pred_peaks) == 0 or len(gt_peaks) == 0:
            continue
        
        # Check the top prediction against all GT centers
        pred_y, pred_x = pred_peaks[0]
        
        pred_angle = np.degrees(
            np.arctan2(pred_sin[i, pred_y, pred_x].item(),
                       pred_cos[i, pred_y, pred_x].item()) / 2.0
        )
        
        matched = False
        for gt_y, gt_x in gt_peaks:
            dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            
            if dist <= center_thresh:
                gt_angle = np.degrees(
                    np.arctan2(gt_sin[i, gt_y, gt_x].item(),
                               gt_cos[i, gt_y, gt_x].item()) / 2.0
                )
                angle_err = abs(pred_angle - gt_angle)
                # Handle angle wrapping (grasps at 0 and 180 are equivalent)
                angle_err = min(angle_err, 180.0 - angle_err)
                
                if angle_err < angle_thresh:
                    matched = True
                    break
        
        if matched:
            successes += 1
    
    return successes / max(batch_size, 1)


def compute_f1_score(pred_heatmap, gt_heatmap, 
                     center_thresh=10, heatmap_thresh=0.3):
    """
    F1-Score for multi-object grasp detection.
    
    Precision = TP / (TP + FP)  — of the grasps we predicted, how many were correct?
    Recall    = TP / (TP + FN)  — of the true grasps, how many did we find?
    F1        = 2 * P * R / (P + R)
    
    A predicted peak is a True Positive if it is within `center_thresh` pixels
    of any ground truth peak.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    batch_size = pred_heatmap.shape[0]
    
    for i in range(batch_size):
        pred_peaks = extract_grasp_centers(pred_heatmap[i], threshold=heatmap_thresh)
        gt_peaks   = extract_grasp_centers(gt_heatmap[i], threshold=heatmap_thresh)
        
        gt_matched = [False] * len(gt_peaks)
        
        for py, px in pred_peaks:
            found_match = False
            for j, (gy, gx) in enumerate(gt_peaks):
                dist = np.sqrt((px - gx)**2 + (py - gy)**2)
                if dist <= center_thresh and not gt_matched[j]:
                    gt_matched[j] = True
                    found_match = True
                    break
            
            if found_match:
                total_tp += 1
            else:
                total_fp += 1
        
        total_fn += sum(1 for m in gt_matched if not m)
    
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return f1, precision, recall


# ==============================================================================
# Training Loop
# ==============================================================================

def train_model(data_dir, num_epochs=60, batch_size=4, lambda_reg=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the DataLoaders
    try:
        train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return

    # 2. Setup the DeepLabV3+ Model
    model = GraspHeatmapNet(pretrained=True).to(device)
    
    # Freeze early encoder layers (conv1, bn1, layer1, layer2) — these learn generic features
    # that transfer well and don't need retraining on our small 96-image dataset
    for name, param in model.encoder.named_parameters():
        # model.encoder is Sequential: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        # Indices 0-5 correspond to conv1, bn1, relu, maxpool, layer1, layer2
        # We freeze parameters belonging to these early stages
        layer_idx = None
        for i, submodule in enumerate(model.encoder):
            for p in submodule.parameters():
                if p is param:
                    layer_idx = i
                    break
            if layer_idx is not None:
                break
        if layer_idx is not None and layer_idx <= 5:  # Freeze up to and including layer2
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Model loaded: {total_params:,} total | {trainable_params:,} trainable | {frozen_params:,} frozen")

    # 3. Optimization setup — differential learning rates
    # Decoder + ASPP + heads get higher LR since they're randomly initialized
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.aspp.parameters()) + \
                     list(model.low_level_conv.parameters()) + \
                     list(model.decoder.parameters()) + \
                     list(model.heatmap_head.parameters()) + \
                     list(model.angle_head.parameters()) + \
                     list(model.width_head.parameters())
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 1e-5},   # Fine-tune unfrozen encoder gently
        {'params': decoder_params, 'lr': 1e-3},    # Train decoder aggressively
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Heatmap loss: MSE for the Gaussian graspability channel
    mse_criterion = nn.MSELoss()
    
    # GradScaler for Mixed Precision Training on RTX 4060
    scaler = GradScaler()

    best_val_loss = float('inf')
    
    # Metrics history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'miou': [], 'success_rate': [], 
        'f1': [], 'precision': [], 'recall': []
    }
    
    # 4. Training Loop
    print("Starting training...")
    print("-" * 100)
    torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        for images, targets in train_loader:
            images  = images.to(device)   # (B, 3, 320, 320)
            targets = targets.to(device)  # (B, 4, 320, 320)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(images)  # (B, 4, 320, 320)
                
                # Split channels
                pred_heatmap = outputs[:, 0, :, :]
                pred_sin     = outputs[:, 1, :, :]
                pred_cos     = outputs[:, 2, :, :]
                pred_width   = outputs[:, 3, :, :]
                
                gt_heatmap = targets[:, 0, :, :]
                gt_sin     = targets[:, 1, :, :]
                gt_cos     = targets[:, 2, :, :]
                gt_width   = targets[:, 3, :, :]
                
                # Mask: only compute regression loss where ground-truth grasps exist
                mask = gt_heatmap > 0.01
                
                # Heatmap loss (entire spatial map)
                loss_heatmap = mse_criterion(pred_heatmap, gt_heatmap)
                
                # Masked regression losses (only at grasp regions)
                loss_sin   = masked_smooth_l1(pred_sin,   gt_sin,   mask)
                loss_cos   = masked_smooth_l1(pred_cos,   gt_cos,   mask)
                loss_width = masked_smooth_l1(pred_width, gt_width, mask)
                
                # Combined loss
                total_loss = loss_heatmap + lambda_reg * (loss_sin + loss_cos + loss_width)
            
            # Scale the loss and call backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += total_loss.item()
            train_batches += 1
            
        epoch_train_loss = running_train_loss / max(train_batches, 1)
        
        # --- Validation Phase with Scientific Metrics ---
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        
        # Accumulators for metrics
        epoch_miou = []
        epoch_sr = []
        epoch_f1_tp = 0
        epoch_f1_fp = 0
        epoch_f1_fn = 0
        
        with torch.inference_mode():
            for images, targets in val_loader:
                images  = images.to(device)
                targets = targets.to(device)
                
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    
                    pred_heatmap = outputs[:, 0, :, :]
                    pred_sin     = outputs[:, 1, :, :]
                    pred_cos     = outputs[:, 2, :, :]
                    pred_width   = outputs[:, 3, :, :]
                    
                    gt_heatmap = targets[:, 0, :, :]
                    gt_sin     = targets[:, 1, :, :]
                    gt_cos     = targets[:, 2, :, :]
                    gt_width   = targets[:, 3, :, :]
                    
                    mask = gt_heatmap > 0.01
                    
                    loss_heatmap = mse_criterion(pred_heatmap, gt_heatmap)
                    loss_sin   = masked_smooth_l1(pred_sin,   gt_sin,   mask)
                    loss_cos   = masked_smooth_l1(pred_cos,   gt_cos,   mask)
                    loss_width = masked_smooth_l1(pred_width, gt_width, mask)
                    
                    total_loss = loss_heatmap + lambda_reg * (loss_sin + loss_cos + loss_width)
                    
                running_val_loss += total_loss.item()
                val_batches += 1
                
                # --- Compute scientific metrics on this batch ---
                # Move to CPU for metric computation
                ph = pred_heatmap.float().cpu()
                gh = gt_heatmap.float().cpu()
                ps = pred_sin.float().cpu()
                pc = pred_cos.float().cpu()
                gs = gt_sin.float().cpu()
                gc = gt_cos.float().cpu()
                
                # Mean IoU
                batch_miou = compute_mean_iou(ph, gh, threshold=0.3)
                epoch_miou.append(batch_miou)
                
                # Success Rate
                batch_sr = compute_success_rate(ph, ps, pc, gh, gs, gc)
                epoch_sr.append(batch_sr)
                
                # F1-Score components (accumulate TP/FP/FN across all batches)
                for i in range(ph.shape[0]):
                    pred_peaks = extract_grasp_centers(ph[i], threshold=0.3)
                    gt_peaks   = extract_grasp_centers(gh[i], threshold=0.3)
                    
                    gt_matched = [False] * len(gt_peaks)
                    for py, px in pred_peaks:
                        found = False
                        for j, (gy, gx) in enumerate(gt_peaks):
                            if np.sqrt((px-gx)**2 + (py-gy)**2) <= 10 and not gt_matched[j]:
                                gt_matched[j] = True
                                found = True
                                break
                        if found:
                            epoch_f1_tp += 1
                        else:
                            epoch_f1_fp += 1
                    epoch_f1_fn += sum(1 for m in gt_matched if not m)
                    
        epoch_val_loss = running_val_loss / max(val_batches, 1)
        
        # Compute epoch-level metrics
        mean_iou = np.mean(epoch_miou) if epoch_miou else 0.0
        success_rate = np.mean(epoch_sr) if epoch_sr else 0.0
        
        precision = epoch_f1_tp / max(epoch_f1_tp + epoch_f1_fp, 1)
        recall    = epoch_f1_tp / max(epoch_f1_tp + epoch_f1_fn, 1)
        f1_score  = 2 * precision * recall / max(precision + recall, 1e-8)
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['miou'].append(mean_iou)
        history['success_rate'].append(success_rate)
        history['f1'].append(f1_score)
        history['precision'].append(precision)
        history['recall'].append(recall)
        
        # Step the LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print comprehensive epoch report
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {epoch_train_loss:.4f} / {epoch_val_loss:.4f} (train/val) | "
              f"mIoU: {mean_iou:.3f} | SR: {success_rate:.3f} | "
              f"F1: {f1_score:.3f} (P={precision:.3f} R={recall:.3f}) | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if epoch_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = epoch_val_loss
            save_path = 'best_grasp_heatmap.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved to {save_path}")
    
    # Save metrics history for paper plots
    print("-" * 100)
    print("Training complete. Final metrics summary:")
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print(f"  Final mIoU:       {history['miou'][-1]:.3f}")
    print(f"  Final SR:         {history['success_rate'][-1]:.3f}")
    print(f"  Final F1:         {history['f1'][-1]:.3f}")
    
    # Save history to disk for the visualization script
    torch.save(history, 'training_history.pth')
    print("  Metrics history saved to training_history.pth")

if __name__ == "__main__":
    # Point this to your processed data folder
    DATA_DIRECTORY = r"d:\Grasp_-CNN\processed_data"
    
    # batch_size=4 is safe for RTX 4060 8GB with DeepLabV3+
    train_model(data_dir=DATA_DIRECTORY, num_epochs=60, batch_size=4)
