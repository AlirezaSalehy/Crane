import torch
import numpy as np
import random
import os
import argparse
import hashlib
import humanhash
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Utility shortcuts exposed at the package level

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA 10.2+
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = 111 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def turn_gradient_off(model):
    print("Turning off gradients in both the image and the text encoder")
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    model.eval()
    return model

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_human_readable_name(args, exclude=['model_name', 'dataset', 'data_path', 'datasets_root_dir',
                                            'checkpoint_path', 'training_path', 'Timestamp', 'why', 
                                            'metrics', 'devices', 'epoch', 'visualize', 'help', None]):
    args=vars(args)
    name_value_pairs = [
        f"{k}_{v}"
        for k,v in args.items()
        if k not in exclude # Exclude "help" or invalid arguments
    ]   
    combined = ",".join(sorted(name_value_pairs))  # Sorting ensures consistent order
    hash_value = hashlib.sha256(combined.encode()).hexdigest()
    human_hash = humanhash.humanize(hash_value, words=2)
    return human_hash

def check_args_conformance_with_train_args(args, training_path):
    # Check if args.txt exists in the training_path
    args_file_path = os.path.join(training_path, 'args.txt')
    configurations_dict = {}  # Dictionary to store configurations
    last_config = {}
    mismatch_descriptions = []  # List to store mismatch descriptions
    if os.path.exists(args_file_path):
        with open(args_file_path, 'r') as f:
            # Read the entire content of the file
            file_content = f.read().strip()
        
        # Split the content into different configurations based on the 'Timestamp' keyword
        configurations = file_content.split('Timestamp:')
        
        # Iterate over each configuration to populate the dictionary
        for config in configurations:
            if config.strip():
                # Convert the configuration to a dictionary
                file_args_dict = {}
                for line in config.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        file_args_dict[key.strip()] = value.strip()
                
                # Store the configuration in the dictionary with a unique key
                timestamp = file_args_dict.get('Timestamp', 'Unknown')
                configurations_dict[timestamp] = file_args_dict
        
        # Select the last configuration for comparison
        if configurations_dict:
            last_timestamp = max(configurations_dict.keys())
            last_config = configurations_dict[last_timestamp]

            # Compare with current args
            print(f"Checking configuration with the most recent timestamp: {last_timestamp}")
            non_critical_mismatches = {
                'dataset', 'device', 'log_dir', 'model_name','dataset_category', 'use_scorebase_pooling', 'aug_rate',\
                'features_list', 'train_with_img_cls_type', 'epoch', 'type', 'save_path', 'train_with_img_cls_prob', 'why',
            }
            if args.dataset != last_config['dataset']:
                non_critical_mismatches.update(['k_shot', 'portion'])
            for key, value in vars(args).items():
                if key in last_config:
                    if str(value) != last_config[key] and key not in non_critical_mismatches:
                        description = f"Argument mismatch for {key}: {value}, but file has {last_config[key]}"
                        print(description)
                        mismatch_descriptions.append(description)
                else:
                    description = f"Argument {key} not found in the most recent args.txt configuration"
                    print(description)
        else:
            print("No valid configuration found in args.txt")
    else:
        print(f"No args.txt file found in {training_path}")
        
    return last_config, mismatch_descriptions

class CustomTensorDataset(Dataset):
    def __init__(self, dataset_features, paths):
        self.dataset = TensorDataset(*dataset_features)
        self.img_paths = paths
        self.length = len(self.dataset)
        
        assert len(self.dataset) == len(self.img_paths), \
        "Number of images and paths must be the same."

    def __getitem__(self, index):
        labels, cls_ids, image_features, patch_features, abnorm_masks = self.dataset[index]
        sample = {
            'anomaly': labels,
            'cls_id': cls_ids, 
            'image_features': image_features,
            'patch_features': patch_features,
            'abnorm_mask': abnorm_masks,
            'img_path': self.img_paths[index]
        }
        
        return sample

    def __len__(self):
        return self.length

def prepare_encode_image_module(model, features_list):
    class EncodeImageModule(torch.nn.Module):
        def __init__(self, model, features_list):
            super(EncodeImageModule, self).__init__()
            self.model = model
            self.features_list = features_list

        def forward(self, image):
            image_features, patch_features = self.model.encode_image(image, self.features_list, self_cor_attn_layers=20)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # patch_features = [patch_feature / patch_feature.norm(dim=-1, keepdim=True) for patch_feature in patch_features]
            patch_features = torch.stack(patch_features, dim=1) 
            return image_features, patch_features
    
    encode_image_module = EncodeImageModule(model, features_list)
    encode_image_module = torch.nn.DataParallel(encode_image_module)
    encode_image_module.cuda()
    return encode_image_module

def precompute_image_features(data, encode_image_module, args):
    batch_size = 2 if args.dino_model == 'dino' else 8
    batch_size *= torch.cuda.device_count()
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=torch.cuda.device_count(),\
                                prefetch_factor=2, pin_memory=True, generator=g, worker_init_fn=seed_worker)
    print(f"Total samples to process: {len(test_dataloader) * test_dataloader.batch_size}")

    device = 'cuda'
    data_items = [[] for _ in range(5)]
    img_paths = []
    for items in tqdm(test_dataloader):
        image = items['img'].to(device) 
        label = items['anomaly']
        cls_id = items['cls_id']
        abnorm_mask = items['abnorm_mask']
        path = items['img_path']

        with torch.no_grad():
            image_features, patch_features = encode_image_module(image)
        
        for index, item in enumerate((label, cls_id, image_features, patch_features, abnorm_mask)):
            data_items[index].append(item.cpu())
        img_paths.extend(path)
        
    data_items = [torch.cat(item_list, dim=0) for item_list in data_items]
    return (data_items, img_paths)