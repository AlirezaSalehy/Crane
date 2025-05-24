import models
from models import Crane
from models.prompt_ensemble import PromptLearner
from dataset.dataset import Dataset
from __init__ import DATASETS_ROOT

from utils.transform import get_transform
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.logger import get_logger
from utils.similarity import calc_similarity_logits, regrid_upsample
from utils import (
    setup_seed,
    seed_worker,
    turn_gradient_off,
    str2bool,
    prepare_encode_image_module,
    precompute_image_features,
    CustomTensorDataset
)

import sys
import os
import argparse
import subprocess

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# import torch.profiler

def train(args):
    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    train_data = Dataset(roots=args.train_data_path, transform=preprocess, 
                        target_transform=target_transform, dataset_name=args.dataset, kwargs=args)
    g = torch.Generator()
    g.manual_seed(args.seed)
    # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # More basic for FPS comparison
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=16, pin_memory=True, prefetch_factor=2,
                                               generator=g, worker_init_fn=seed_worker)  # Faster
    print(f"Length of the dataset: {len(train_data)}")

    ##########################################################################################
    device = 'cuda' if torch.cuda.is_available() else "cpu"    
    print(device)

    crane_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx, 'others': args}    
    model, _ = models.load("ViT-L/14@336px", device=device, design_details = crane_parameters)
    model = turn_gradient_off(model) 
    model.visual.replace_with_EAttn(to_layer=20, type=args.attn_type) # Replace last 20 layers
    if args.dino_model != 'none':
        model.use_DAttn(args.dino_model)
        
    prompt_learner = PromptLearner(model.to("cpu"), crane_parameters)
    sbp = Crane.ScoreBasePooling()

    model.to(device)
    prompt_learner.to(device)

    ##########################################################################################
    params = list(prompt_learner.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.6, 0.999))
    
    precompute = False
    if precompute:
        encode_image_module = prepare_encode_image_module(model, args.features_list)
        precompute_features, pathes = precompute_image_features(train_data, encode_image_module, args)
        precompute_dataset = CustomTensorDataset(precompute_features, pathes)
        train_dataloader = DataLoader(precompute_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                       generator=g, worker_init_fn=seed_worker)
        model.visual.to('cpu')
    
    # losses
    ce_loss_focal = FocalLoss() 
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        loss_list = []

        with tqdm(train_dataloader) as batch_tqdm:
            for items in batch_tqdm:
                label =  items['anomaly'].to(device)
                abnorm_mask = items['abnorm_mask'].squeeze().to(device)
                
                if precompute:
                    image_features, patch_features = items['image_features'].to(device), items['patch_features'].to(device)
                    patch_features = patch_features.permute(1, 0, *range(2, patch_features.dim())) # 4, N, L, C
                else:
                    image = items['img'].to(device)
                    image_features, patch_features = model.encode_image(image, args.features_list, self_cor_attn_layers=20)
                    patch_features = torch.stack(patch_features, dim=0) 
                image_features = F.normalize(image_features, dim=-1) 
                patch_features = F.normalize(patch_features, dim=-1)
            
                # Text Features
                #########################################################################
                prompts, tokenized_prompts, compound_prompts_text, is_train_with_img_cls = prompt_learner(img_emb=image_features)

                if is_train_with_img_cls: 
                    text_features_nrm = model.encode_text_learn(prompts[0], tokenized_prompts[0], compound_prompts_text) # input dims: 2, 77 | 2, 77, 768 | 9, 4, 768
                    text_features_anm = model.encode_text_learn(prompts[1], tokenized_prompts[1], compound_prompts_text) # input dims: 2, 77 | 2, 77, 768 | 9, 4, 768
                    text_features = torch.stack([text_features_nrm, text_features_anm], dim=1)
                else:
                    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).unsqueeze(dim=0) # 2, 77 | 2, 77, 768 | 9, 4, 768
                text_features = F.normalize(text_features, dim=-1).float() # 1, 2, 768

                # Similarity Map - Segmentation
                #########################################################################
                similarity_map_list = []
                for patch_feature in patch_features: 
                    pixel_logits = calc_similarity_logits(patch_feature, text_features, temp=0.07)
                    pixel_scores = pixel_logits.softmax(dim=-1)
                    similarity_map = regrid_upsample(pixel_scores, args.image_size, mode=args.interpolation) 
                    similarity_map_list.append((similarity_map, pixel_logits))

                ce_focal_loss = 0
                dice_loss = 0
                for i in range(len(similarity_map_list)):
                    whole_map = (1-similarity_map_list[i][0][...,0] + similarity_map_list[i][0][...,1])/2
                    smlr_map = similarity_map_list[i][0].permute(0, 3, 1, 2) 
 
                    dice_loss += loss_dice(whole_map, abnorm_mask)
                    ce_focal_loss += ce_loss_focal(smlr_map, abnorm_mask)
                    
                # Similarity Score - Classification
                #########################################################################
                if args.use_scorebase_pooling: 
                    alpha = 0.5
                    sms = [sm_lst[1] for sm_lst in similarity_map_list]
                    clustered_feature = sbp.forward(patch_features, sms) 
                    image_features = alpha * clustered_feature + (1 - alpha) * image_features # aggregates the class token and the clustered features for more comprehensive information
                    image_features = F.normalize(image_features, dim=1)

                image_logits = calc_similarity_logits(image_features, text_features, temp=0.01) # batch_size, 1, 768 @ batch_size, 768, 2 or 3
                ce_img2txt_loss = F.cross_entropy(image_logits, label.long().to(device)) 
                # txt2img_lbl = torch.stack([(1-label), label], dim=0)/label.sum()
                # ce_txt2img_loss = F.cross_entropy(image_logits.permute(1, 0), txt2img_lbl.to(device))                                         

                #loss
                optimizer.zero_grad()
                dice_loss *= 2
                ce_focal_loss *= 2
                ls = ce_focal_loss+dice_loss+0.2*ce_img2txt_loss
                ls.backward() 
                optimizer.step()

                loss_list.append((ce_focal_loss.item(), dice_loss.item(), ce_img2txt_loss.item()))
                batch_tqdm.set_description(f"ce_fc_ls: {ce_focal_loss:.3f}, bcd_dice_ls: {dice_loss:.3f}, ce_img_ls: {ce_img2txt_loss:.3f}")                
        # logs
        ce_focal_ls, dice_ls, ce_img_ls = np.mean(loss_list, axis=0)
        log_template = 'epoch [{}/{}], ce_fc_ls:{:.4f}, bdc_ls:{:.4f}, ce_img_ls:{:.4f}'
        logger.info(log_template.format(epoch + 1, args.epoch, ce_focal_ls, dice_ls, ce_img_ls))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            prmtp_ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            checkpoint_data = {"prompt_learner": prompt_learner.state_dict()}
            torch.save(checkpoint_data, prmtp_ckp_path)

if __name__ == '__main__':
    dss = ['mvtec']
    
    parser = argparse.ArgumentParser("Crane", add_help=True)
    parser.add_argument("--datasets_root_dir", type=str, default=f"{DATASETS_ROOT}")
    parser.add_argument("--train_data_path", type=str, nargs="+", default=[f"{DATASETS_ROOT}/{ds}/" for ds in dss])
    parser.add_argument("--save_path", type=str, default='./checkpoints/')
    parser.add_argument("--model_name", type=str, default="default") # NOTE: The "trained_on_<DATASET_NAME>" will be prepended to the model name for saving checkpoints
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")

    parser.add_argument("--type", type=str, default='train') 
    parser.add_argument("--device", type=int, default=0, help="cuda device")
    parser.add_argument("--epoch", type=int, default=5, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--aug_rate", type=float, default=0.0, help="augmentation rate")

    parser.add_argument("--dataset", type=str, nargs="+", default=[f'{ds}' for ds in dss], help="train dataset name")
    parser.add_argument("--k_shot", type=int, default=0, help="samples per class for few-shot learning. 0 means use all data.")
    parser.add_argument("--portion", type=float, default=1) 

    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="layer features used")
    parser.add_argument("--interpolation", type=str, choices=['nearest', 'bilinear'], default='nearest') 

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
     
    parser.add_argument("--train_with_img_cls_prob", type=float, default=1)
    parser.add_argument("--train_with_img_cls_type", type=str, choices=["none", "replace_prefix", "replace_suffix", "pad_prefix", "pad_suffix"], default="pad_suffix")
    parser.add_argument("--dino_model", type=str, choices=['none', 'dinov2', 'dino', 'sam'], default='dinov2')
    parser.add_argument("--both_eattn_dattn", type=str2bool, default=True)
    parser.add_argument("--use_scorebase_pooling", type=str2bool, default=True) 
    parser.add_argument("--attn_type", type=str, choices=["vv", "kk", "qq", "qq+kk", "qq+kk+vv", "(q+k+v)^2"], default="qq+kk+vv")
    parser.add_argument("--why", type=str, default="Neccessity of the experiment")

    args = parser.parse_args()
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device) 
        command = [sys.executable, __file__, ] + sys.argv[1:]  
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        setup_seed(args.seed)        

        # paths
        args.train_data_path = [f"{args.datasets_root_dir}/{ds}/" for ds in args.dataset]
        args.save_path = f'{args.save_path}/trained_on_{"_".join(args.dataset)}_{args.model_name}/'
        print(f'running {args.model_name}')

        train(args)