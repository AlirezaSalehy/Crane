import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from models import Crane
from models.prompt_ensemble import PromptLearner
from dataset.dataset import Dataset
from __init__ import DATASETS_ROOT

from utils.transform import get_transform
from utils.visualization import visualizer
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.logger import get_logger, save_args_to_file
from utils.similarity import calc_similarity_logits, regrid_upsample
from utils import (
    setup_seed,
    turn_gradient_off,
    str2bool,
    make_human_readable_name,
)

from scipy.ndimage import gaussian_filter
import pandas as pd 
import sys
import os
import subprocess
import argparse
from tqdm import tqdm
from tabulate import tabulate

from termcolor import colored

class ScoreCalculator(nn.Module):
    def __init__(self, base_model, class_details, args,
                 prompt_learner, score_base_pooling):
        super().__init__()
        self.model = base_model
        self.class_details = class_details
        self.args = args

        self.prompt_learner = prompt_learner
        self.sbp = score_base_pooling
        self._cached_text = {}

    def forward(self, image):
        with torch.no_grad():
            image_features, patch_list = self.model.encode_image(image, self.args.features_list, self_cor_attn_layers=20)
            patch_features = torch.stack(patch_list, dim=0)
        image_features = F.normalize(image_features, dim=-1)
        patch_features = F.normalize(patch_features, dim=-1)
        
        if self.args.train_with_img_cls_prob != 0:
            with torch.no_grad():
                prompts, tokenized_prompts, compound_prompts_text, is_train_with_img_cls = \
                    self.prompt_learner(img_emb=image_features)
                if is_train_with_img_cls:
                    text_features_nrm = self.model.encode_text_learn(prompts[0], tokenized_prompts[0], compound_prompts_text).float()
                    text_features_anm = self.model.encode_text_learn(prompts[1], tokenized_prompts[1], compound_prompts_text).float()
                    text_features = torch.stack([text_features_nrm, text_features_anm], dim=1)
                else:
                    text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).unsqueeze(dim=0).float()
                text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = self.text_features.to(image.device)

        # Similarity Map - Segmentation
        #########################################################################
        pixel_logits_list = []
        for patch_feature in patch_features:
            pixel_logits = calc_similarity_logits(patch_feature, text_features)
            pixel_logits_list.append(pixel_logits)

        if self.args.soft_mean:    
            similarity_maps = [regrid_upsample(pl.softmax(dim=-1), args.image_size) for pl in pixel_logits_list]
            score_map = torch.stack(similarity_maps).mean(dim=0)
        else:
            logits_maps = [regrid_upsample(pl, args.image_size) for pl in pixel_logits_list]
            mean_logits_map = torch.stack(logits_maps).mean(dim=0)
            score_map = mean_logits_map.softmax(dim=-1)
        anomaly_map = score_map[..., 1]

        # Classification Score
        #########################################################################
        if self.args.use_scorebase_pooling:
            alpha = 0.5
            clustered_feature = self.sbp.forward(patch_features, pixel_logits_list)
            image_features = alpha * clustered_feature + (1 - alpha) * image_features
            image_features = F.normalize(image_features, dim=1)

        image_logits = calc_similarity_logits(image_features, text_features)
        image_pred = image_logits.softmax(dim=-1)
        anomaly_score = image_pred[:, 1].detach()

        return anomaly_score, anomaly_map 

def compute_metrics_for_object(obj, dataset_results):
    dataset_results[obj]['imgs_masks'] = torch.stack(dataset_results[obj]['imgs_masks'])
    dataset_results[obj]['anomaly_maps'] = torch.stack(dataset_results[obj]['anomaly_maps'])
    
    image_auroc = image_level_metrics(dataset_results, obj, "image-auroc")
    image_ap = image_level_metrics(dataset_results, obj, "image-ap")
    image_f1 = image_level_metrics(dataset_results, obj, "image-f1")

    pixel_auroc = pixel_level_metrics(dataset_results, obj, "pixel-auroc")
    pixel_aupro = pixel_level_metrics(dataset_results, obj, "pixel-aupro")
    pixel_f1 = pixel_level_metrics(dataset_results, obj, "pixel-f1")

    dataset_results[obj] = None
    return {
        "pixel_auroc": pixel_auroc,
        "pixel_aupro": pixel_aupro,
        "pixel_f1": pixel_f1,
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "image_f1": image_f1,
    }

def process_dataset(model, dataloader, class_details, args): 
    Crane_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx, 'others': args}
    prompt_learner = PromptLearner(model.to('cpu'), Crane_parameters)  
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    missing_keys, unexpected_keys = prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
    assert len(missing_keys) == 0, f"Missing keys in state dict: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys in state dict: {unexpected_keys}"
    
    score_base_pooling = Crane.ScoreBasePooling()

    score_calc = ScoreCalculator(
        model, class_details, args,
        prompt_learner=prompt_learner,
        score_base_pooling=score_base_pooling
    )
    
    if args.train_with_img_cls_prob == 0:
        with torch.no_grad():
            prompts, tokenized_prompts, compound_prompts_text, _ = prompt_learner()
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = F.normalize(text_features, dim=-1).cuda()
        score_calc.text_features = text_features  

    dp_calc = nn.DataParallel(score_calc, device_ids=args.devices)
    dp_calc.eval()
    dp_calc.cuda()
    
    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': [], 'img_paths': []}
            for obj in class_details[1]}
    for items in tqdm(dataloader, desc="Processing test samples"):
        anomaly_score, anomaly_map = dp_calc(items['img'].cuda())
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)

        for i in range(items['abnorm_mask'].size(0)):
            inst_cls = items['cls_id'][i].item()
            results[inst_cls]['anomaly_maps'].append(anomaly_map[i].cpu())
            results[inst_cls]['pr_sp'].append(anomaly_score[i].cpu())
            results[inst_cls]['imgs_masks'].append(items['abnorm_mask'][i].squeeze(0))
            results[inst_cls]['gt_sp'].append(items['anomaly'][i])
            results[inst_cls]['img_paths'].append(items['img_path'][i])
    
    torch.cuda.empty_cache()
        
    class_names, class_ids = class_details
    if args.visualize:
        for clss, dic in results.items():
            visualizer(dic['img_paths'], dic['anomaly_maps'], dic['imgs_masks'], 518, f'{args.model_name}/{args.dataset}/{args.log_dir}/{class_names[clss]}', draw_contours=True)

    epoch_metrics = []
    for obj_id in class_ids:
        print(f'calculating metrics for {class_names[obj_id]}')
        class_metrics = compute_metrics_for_object(obj_id, results)
        class_metrics['objects'] = class_names[obj_id]
        epoch_metrics.append(class_metrics)
        
    return epoch_metrics

def generate_epoch_performance_table(epoch_metrics_dataframe, class_names):
    epoch_metrics_dataframe = pd.DataFrame(epoch_metrics_dataframe).set_index('objects')
    epoch_metrics_dataframe = epoch_metrics_dataframe.loc[class_names] # Sort
    epoch_metrics_dataframe.loc['mean'] = epoch_metrics_dataframe.mean()
    results = tabulate(epoch_metrics_dataframe, headers='keys', tablefmt="pipe", floatfmt=".03f")
    return results

def evaluate(model, items, class_details, args):
    save_path = f'{args.save_path}/{args.log_dir}/epoch_{args.epoch}/'
    logger = get_logger(save_path)

    print(f"process_dataset, Batch size: {args.batch_size}")
    # dataloader = DataLoader(items, batch_size=args.batch_size, shuffle=False)
    dataloader = DataLoader(items, batch_size=args.batch_size, shuffle=False, num_workers=8, prefetch_factor=2, pin_memory=True,)
    epoch_metrics = process_dataset(model, dataloader, class_details, args)
    epoch_report = generate_epoch_performance_table(epoch_metrics, class_details[0])
    
    print(args.dataset)
    logger.info("\n%s", epoch_report)
    
def test(args):        
    Crane_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,\
                                "learnabel_text_embedding_length": args.t_n_ctx, 'others': args}
    model, _ = models.load("ViT-L/14@336px", device='cuda', design_details=Crane_parameters)
    model.visual.replace_with_EAttn(to_layer=20, type=args.attn_type)
    if args.dino_model != 'none':
        model.use_DAttn(args.dino_model)
    model = turn_gradient_off(model)

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(roots=args.data_path, transform=preprocess, target_transform=target_transform, \
                                dataset_name=args.dataset, kwargs=args)
    class_details = (test_data.cls_names, test_data.class_ids)

    evaluate(model, test_data, class_details, args)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser("Crane", add_help=True)
    # model
    parser.add_argument("--model_name", type=str, default="trained_on_mvtec_default", help="model_name")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--visualize", type=str2bool, default=False)
    
    parser.add_argument("--type", type=str, default='test') 
    parser.add_argument("--devices", nargs='+', type=int, default=[0])
    parser.add_argument("--epoch", type=int, default=5) 
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--aug_rate", type=float, default=0.0, help="augmentation rate")

    parser.add_argument("--datasets_root_dir", type=str, default=f"{DATASETS_ROOT}")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--portion", type=float, default=1) # 0.02
    parser.add_argument("--k_shot", type=int, default=0, help="number of samples per class. 0 means use all data.")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")    
    parser.add_argument("--train_with_img_cls_prob", type=float, default=0)
    parser.add_argument("--train_with_img_cls_type", type=str, choices=["none", "replace_prefix", "replace_suffix", "pad_prefix", "pad_suffix"], default="pad_suffix")

    parser.add_argument("--dino_model", type=str, choices=['none', 'dinov2', 'dino', 'sam'], default='dinov2')
    parser.add_argument("--use_scorebase_pooling", type=str2bool, default=True)

    parser.add_argument("--image_size", type=int, default=518, help="input image size")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="layer features used")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--soft_mean", type=str2bool, default=False) 

    parser.add_argument("--attn_type", type=str, choices=["vv", "kk", "qq", "qq+kk", "qq+kk+vv", "(q+k+v)^2"], default="qq+kk+vv")
    parser.add_argument("--both_eattn_dattn", type=str2bool, default=True)

    parser.add_argument("--why", type=str, help="Explanation about this experiment and how it is different other than parameter values")
    args = parser.parse_args()
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ: # Forcing all the tensors to be on the specified device(s)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.devices)) if len(args.devices) > 1 else str(args.devices)
        command = [sys.executable,  __file__, ] + sys.argv[1:] 
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        setup_seed(args.seed)
        args.log_dir = make_human_readable_name(args)        
        
        args.data_path = [f"{args.datasets_root_dir}/{args.dataset}/"]
        args.checkpoint_path = f'./checkpoints/{args.model_name}/epoch_{args.epoch}.pth'
        args.save_path = f'./results/{args.model_name}/test_on_{args.dataset}/'

        print(f"Testing on dataset from: {args.data_path}") 
        print(f"Results will be saved to: {colored(args.save_path+args.log_dir, 'green')}")

        save_args_to_file(args, sys.argv[1:], log_dir=args.log_dir)

        test(args)
        