import cv2
import os
from utils.transform import normalize
import numpy as np
import torch

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def denormalize(tensor, mean, std):
    # Convert mean and std to tensors for broadcasting
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    # Denormalize the tensor: (value * std) + mean
    denormalized_tensor = tensor * std + mean
    return denormalized_tensor

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    # Convert scoremap from a PyTorch tensor to a NumPy array
    if isinstance(scoremap, torch.Tensor):
        scoremap = scoremap.detach().cpu().numpy()  # Convert tensor to NumPy array
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualizer(pathes, anomaly_map, masks, img_size, cls_name, save_path='./vis_img/', draw_contours=True):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        
        # Save the final visualization
        save_vis = os.path.join(save_path, 'imgs', str(cls_name), str(cls))
        os.makedirs(save_vis, exist_ok=True)
        
        # Load original image and resize
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)
        filename_orig = filename.split('.')[0] + "_orig." + filename.split('.')[-1]  # Append '_orig'
        cv2.imwrite(os.path.join(save_vis, filename_orig), vis)

        # Use the provided mask (it's guaranteed to be available)
        gt_mask = masks[idx].detach().cpu().numpy()
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255  # Convert to binary (0 or 255)

        # Normalize and apply anomaly map
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)

        # Convert back to BGR for OpenCV
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        filename_pred = filename.split('.')[0] + "_pred." + filename.split('.')[-1]  # Append '_pred'
        cv2.imwrite(os.path.join(save_vis, filename_pred), vis)
        
        # Find and overlay contours (only if draw_contours is True)
        if draw_contours:
            filename_ctr = filename.split('.')[0] + "_cntr." + filename.split('.')[-1]  # Append '_cntr' before file extension
            contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (120, 251, 120), 2)  # Pale green contours
            cv2.imwrite(os.path.join(save_vis, filename_ctr), vis)
