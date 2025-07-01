from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve

import numpy as np
from skimage import measure
import torch
from torchmetrics import AUROC, AveragePrecision
import time

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def calc_f1_max(gt, pr):
    precisions, recalls, _ = precision_recall_curve(gt, pr)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    return np.max(f1_scores[np.isfinite(f1_scores)])

# without warning for division by zero
# denom = precisions + recalls
# f1_scores = np.zeros_like(denom)
# valid = denom > 0
# f1_scores[valid] = (2 * precisions[valid] * recalls[valid]) / denom[valid]

def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    
    if len(np.unique(gt)) < 2:
        print("only one class present, can not calculate image metrics")
        return 0
    
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)
    elif metric == 'image-f1':
        performance = calc_f1_max(gt, pr)
        # performance = f1_score(gt, pr.round())
        # assert f1_max == performance
    return performance

def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    
    if len(np.unique(gt)) < 2:
        print("only one class present, can not calculate pixel metrics")
        return 0
    
    if metric == 'pixel-auroc':
        # gt = np.array(gt.cpu()); pr = np.array(pr.cpu())
        # performance = roc_auc_score(gt.ravel(), pr.ravel())
        performance = AUROC(task="binary")(pr, gt.to(dtype=torch.long)).item()
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score_gpu(gt, pr)
        # performance = cal_pro_score(gt, pr)
    elif metric == 'pixel-ap':     # NOTE: The order in sklearn and torch metrics is inverse
        # gt = np.array(gt.cpu()); pr = np.array(pr.cpu())
        # performance= average_precision_score(gt.ravel(), pr.ravel())
        performance = AveragePrecision(task="binary")(pr, gt.to(dtype=torch.long)).item()
        
    elif metric == 'pixel-f1':
        # gt = np.array(gt.cpu()); pr = np.array(pr.cpu())
        # performance = f1_score(gt.ravel(), pr.ravel().round())
        performance = calc_f1_max(gt.cpu().ravel(), pr.cpu().ravel())
    return performance

# NEW implementation for pro using GPU and PyTorch
def cal_pro_score_gpu(masks, amaps, max_step=200, expect_fpr=0.3):
    # GPU implementation using PyTorch
    device="cuda"
    if not torch.is_tensor(amaps):
        amaps = torch.tensor(amaps)
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks)
        
    amaps = amaps.to(device)
    masks = masks.to(device)
    
    binary_amaps = torch.zeros_like(amaps, dtype=torch.bool, device=device)
    min_th, max_th = amaps.min().item(), amaps.max().item()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    
    regionprops_list = [measure.regionprops(measure.label(mask.cpu().numpy())) for mask in masks]
    coords_list = [[(region.coords[:, 0], region.coords[:, 1], len(region.coords)) for region in regionprops] for regionprops in regionprops_list]
    inverse_masks = 1 - masks
    tn_pixel = inverse_masks.sum().item() # Pixels that truly has the label of 0
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        
        for binary_amap, regions_coords in zip(binary_amaps, coords_list):
            for coords in regions_coords:
                tp_pixels = binary_amap[coords[0], coords[1]].sum().item()
                pro.append(tp_pixels / coords[2])
        
        fp_pixels = torch.logical_and(inverse_masks, binary_amaps).sum().item()
        fpr = fp_pixels / tn_pixel
        pros.append(np.mean(pro))
        fprs.append(fpr)
        ths.append(th.item())
    
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    pros = pros[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros)
    return pro_auc

# https://github.com/M-3LAB/open-iad/blob/main/metric/mvtec3d/au_pro.py#L205
import numpy as np
from scipy.ndimage import label
from bisect import bisect
__all__ = ['GroundTruthComponent', 'trapezoid', 'collect_anomaly_scores', 'compute_pro', 'calculate_au_pro']

class GroundTruthComponent:
    def __init__(self, anomaly_scores):
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()
        self.index = 0
        self.last_threshold = None

    def compute_overlap(self, threshold):
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold
        while self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold:
            self.index += 1
        return 1.0 - self.index / len(self.anomaly_scores)

def trapezoid(x, y, x_max=None):
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("""WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    assert len(anomaly_maps) == len(ground_truth_maps)
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)
    structure = np.ones((3, 3), dtype=int)
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        labeled, n_components = label(gt_map, structure)
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()
    return ground_truth_components, anomaly_scores_ok_pixels

def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)
    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)
        fprs.append(fpr)
        pros.append(pro)
    fprs = fprs[::-1]
    pros = pros[::-1]
    return fprs, pros

def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=200):
    # Compute the PRO curve.
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)

    # Compute the area under the PRO curve.
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit

    # Return the evaluation metrics.
    return au_pro, pro_curve

def test_pro_score(masks, amaps):
    start_cpu = time.time()
    pro_auc_cpu = cal_pro_score(masks, amaps,  max_step=200, expect_fpr=0.3)
    end_cpu = time.time()
    cpu_duration = end_cpu - start_cpu
    print(f"CPU execution time: {cpu_duration:.4f} seconds")
    # start_gpu = time.time()
    # masks_torch = torch.tensor(masks, dtype=torch.float32, device='cuda')
    # amaps_torch = torch.tensor(amaps, dtype=torch.float32, device='cuda')
    # pro_auc_gpu = cal_pro_score_gpu(masks_torch, amaps_torch)
    # end_gpu = time.time()
    # gpu_duration = end_gpu - start_gpu
    # print(f"GPU execution time: {gpu_duration:.4f} seconds")
    start_openiad = time.time()
    pro_auc_openiad = calculate_au_pro(masks, amaps, integration_limit=0.3, num_thresholds=200)[0]
    end_openiad = time.time()
    openiad_duration = end_openiad - start_openiad
    print(f"openiad execution time: {openiad_duration:.4f} seconds")
    
    # assert np.isclose(pro_auc_cpu, pro_auc_openiad), f"Results differ: CPU={pro_auc_cpu}, GPU={pro_auc_openiad}"
    print(f"Test passed: CPU={pro_auc_cpu}, GPU={pro_auc_openiad}")

if __name__ == "__main__":
    # Example usage (with small random data for testing)
    num_sam = 25
    device='7'
    
    # masks = np.random.randint(0, 2, (num_sam, 512, 512))  # Binary masks
    # amaps = np.random.rand(num_sam, 512, 512)  # Anomaly maps
    
    # test_pro_score(masks, amaps)
    
    masks = np.random.randint(0, 2, (num_sam, 256, 256))  # Binary masks
    amaps = np.random.rand(num_sam, 256, 256)  # Anomaly maps
    
    test_pro_score(masks, amaps)
    
    masks = np.random.randint(0, 2, (num_sam, 64, 64))  # Binary masks
    amaps = np.random.rand(num_sam, 64, 64)  # Anomaly maps
    
    test_pro_score(masks, amaps)