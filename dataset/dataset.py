import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import os
import albumentations as A

def anomaly_map_guided_crop(img, img_mask):
    # Convert mask to numpy for bounding box calculation
    mask_np = np.array(img_mask)
    if mask_np.sum() > 0:  # Check if there is any anomaly in the mask
        # Get the bounding box of the anomaly
        nonzero_coords = np.column_stack(np.nonzero(mask_np))
        top_left = nonzero_coords.min(axis=0)  # (y_min, x_min)
        bottom_right = nonzero_coords.max(axis=0)  # (y_max, x_max)
        
        # Optionally, expand the bounding box to include some background
        padding = img.size[0]*0.1 + np.random.randint(0, int(img.size[0]-img.size[0]*0.1))  # This can be adjusted based on the context you want to include
        y_min, x_min = np.maximum([0, 0], top_left - padding)
        y_max, x_max = np.minimum([mask_np.shape[0], mask_np.shape[1]], bottom_right + padding)
        
        # Crop both image and mask
        img = img.crop((x_min, y_min, x_max, y_max))
        img_mask = img_mask.crop((x_min, y_min, x_max, y_max))
        
    return img, img_mask

def save_selected_data_paths(data_all, folder_path, file_name='selected_data_paths.txt'):
    file_path = os.path.join(folder_path, file_name)
    
    # Read existing lines from the file
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_lines = f.readlines()
    else:
        existing_lines = []

    # Ensure existing_lines has at least as many lines as data_all
    while len(existing_lines) < len(data_all):
        existing_lines.append('\n')

    # Update lines with new img_path data
    with open(file_path, 'w') as f:
        for i, data in enumerate(data_all):
            if 'img_path' in data:
                existing_line = existing_lines[i].rstrip('\n')
                updated_line = existing_line + ' ' + data['img_path'] if existing_line else data['img_path']
                f.write(updated_line + '\n')
            else:
                print("Warning: 'img_path' not found in data entry.")

def compare_data_with_file(data_list, folder_path, key='img_path'):
    # Construct the full file path
    file_path = os.path.join(folder_path, 'selected_data_paths.txt')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return 0, len(data_list)  # Return 0 matches and all data as mismatches

    # Read the file paths from the file
    with open(file_path, 'r') as f:
        stored_file_paths = [line.strip() for line in f.readlines()]

    # Extract the file paths from the data list
    data_file_paths = [data[key] for data in data_list if key in data]

    # Calculate matches and mismatches
    matches = set(data_file_paths) & set(stored_file_paths)
    mismatches_data = set(data_file_paths) - matches
    mismatches_file = set(stored_file_paths) - matches

    match_count = len(matches)
    mismatch_count = len(mismatches_data) + len(mismatches_file)

    match_description = f"Matches ({match_count}): {matches}"
    mismatch_description = (f"Mismatches in data ({len(mismatches_data)}): {mismatches_data}, "
                            f"Mismatches in file ({len(mismatches_file)}): {mismatches_file}")
    print(match_description)
    print(mismatch_description)

    return match_count, mismatch_count

# Combining MVTec images to form VisA samples
def combine_img(organized_data, random_defect=None): # random_defect = "abnormal"
    img_ls = []
    mask_ls = []
    for i in range(4):
        if random_defect is None:
            # random_defect = random.choice(list(organized_data.keys()))
            random_defect = random.choices(list(organized_data.keys()), weights=[0.8, 0.2], k=1)[0] # With these weight we make sure that normal images are created nrealy 50% of times
        
        random_sample = random.choice(organized_data[random_defect])
         
        img_path = os.path.join(random_sample['root'], random_sample['img_path'])
        mask_path = os.path.join(random_sample['root'], random_sample['mask_path'])
        assert (os.path.exists(img_path))
        img = Image.open(img_path)
        img_ls.append(img)
        if random_sample['anomaly'] == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            assert os.path.exists(mask_path)
            img_mask = np.array(Image.open(mask_path).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        mask_ls.append(img_mask)

    # image
    image_width, image_height = img_ls[0].size
    result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
    for i, img in enumerate(img_ls):
        row = i // 2
        col = i % 2
        x = col * image_width
        y = row * image_height
        result_image.paste(img, (x, y))

    # mask
    result_mask = Image.new("L", (2 * image_width, 2 * image_height))
    for i, img in enumerate(mask_ls):
        row = i // 2
        col = i % 2
        x = col * image_width
        y = row * image_height
        result_mask.paste(img, (x, y))

    return result_image, result_mask

class Dataset(data.Dataset):
    def __init__(self, roots, transform, target_transform, dataset_name, kwargs=None):
        self.roots = roots
        self.transform = transform
        self.target_transform = target_transform
        
        self.aug_rate = kwargs.aug_rate
        pr=0.20 # 0.5 #
        self.img_trans = A.Compose([
			A.Rotate(limit=30, p=pr),
			A.RandomRotate90(p=pr),
			A.RandomBrightnessContrast(p=pr),
			A.GaussNoise(p=pr),
			A.OneOf([
				A.Blur(blur_limit=3, p=pr),
				A.ColorJitter(p=pr),
				A.GaussianBlur(p=pr),
			], p=pr)
		], is_check_shapes=False)
        
        meta_infos = {}
        dataset_split='test'
        for root in roots:
            with open(f'{root}/meta.json', 'r') as f:
                meta_info = json.load(f)
                for cls in meta_info[dataset_split]:
                    meta_info[dataset_split][cls] = [{**s, 'root': root} for s in meta_info[dataset_split][cls]]
                    
                    if cls in meta_infos:
                        meta_infos[cls].extend(meta_info[dataset_split][cls])
                    else:
                        meta_infos[cls] = meta_info[dataset_split][cls]
        self.cls_names = list(meta_infos.keys())
        
        self.data_all = []
        for cls_name in self.cls_names:
            self.data_all.extend(meta_infos[cls_name])

        self.dataset_name = dataset_name
        self.class_ids = list(range(len(self.cls_names)))
        self.class_name_map_class_id = {k: index for k, index in zip(self.cls_names, self.class_ids)}

        self.portion = kwargs.portion
        self.k_shot = kwargs.k_shot
        self.mode = kwargs.type
        if not (self.portion == 1.0 and self.k_shot == 0):
            sampled_sets = self._sample(meta_infos)
            
            if self.mode == 'train':
                self.data_all = sampled_sets[0] # 
                # save_train_data_paths(data_all)
            
            # elif self.mode == 'test' and kwargs.train_dataset == self.dataset_name:
                # self.data_all = sampled_sets[1] 
                # compare_data_with_file(self.data_all, kwargs.training_path)

            # save_selected_data_paths(self.data_all, kwargs.save_path)
        
        
        self.organized_data = {cls_name: {'normal': [], 'abnormal': []} for cls_name in self.cls_names}
        for entry in self.data_all:
            cls_name = entry['cls_name']
            anomaly_status = 'abnormal' if entry['anomaly'] == 1 else 'normal'
            self.organized_data[cls_name][anomaly_status].append(entry)
        
        self.length = len(self.data_all)
        print(f"number of samples: {self.length}")

    def augment(self, img , img_mask):
        img_mask = np.array(img_mask)
        img = np.array(img)
        augmentations = self.img_trans(mask=img_mask, image=img)
        img = augmentations["image"]
        img_mask = augmentations["mask"]
        img = Image.fromarray(img)
        img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
        return img, img_mask

    # Sample same number of normal and anomalous for all classes
    # Or to sample proportional to their length
    def _sample(self, meta_info):
        if self.portion == 1.0 and self.k_shot == 0:
            return self.data_all
        
        sampled_data = []
        complement_data = [] 
        
        for cls_name, data_list in meta_info.items():
            nrm_smpls = [item for item in data_list if item['anomaly'] == 0]
            anm_smpls = [item for item in data_list if item['anomaly'] == 1]
            
            if self.k_shot > 0:
                n_samples = self.k_shot
                n_nrm_smpls = min(int(n_samples/2), len(nrm_smpls))
                n_anm_smpls = min(int(n_samples/2), len(anm_smpls))
                
            else:
                n_nrm_smpls = int(len(nrm_smpls)*self.portion)
                n_anm_smpls = int(len(anm_smpls)*self.portion)

            cls_data = []
            cls_data.extend(random.sample(nrm_smpls, n_nrm_smpls))
            cls_data.extend(random.sample(anm_smpls, n_anm_smpls))
            sampled_data.extend(cls_data)
            
            complement_class_data = [item for item in data_list if item not in cls_data]
            complement_data.extend(complement_class_data)
            
            if self.k_shot + self.portion > 0:
                print(f'num samples for cls {cls_name}, norm: {n_nrm_smpls}, anom: {n_anm_smpls}')
            
        return sampled_data, complement_data

    def __len__(self):
        return self.length

    def _process_image(self, data):
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], \
                                                                data['cls_name'], data['specie_name'], data['anomaly']
        
        root = data['root']    
        img = Image.open(os.path.join(root, img_path))

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            if os.path.isdir(os.path.join(root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        random_number = random.random()
        if self.mode == 'train' and self.aug_rate > random_number and \
            cls_name in ['hazelnut', 'pill', 'zipper', 'bottle', 'screw', 'metal_nut', 'cable']:
            img, img_mask = combine_img(self.organized_data[cls_name])
            anomaly = 1 if np.any(np.array(img_mask) > 0) else 0
        # if self.mode == "train":
        #     self.augment(img=img, img_mask=img_mask)

        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask)
        img_mask[img_mask > 0.5] = 1
        img_mask[img_mask <= 0.5] = 0

        result = {
            'img': img,
            'abnorm_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': os.path.join(root, img_path),
            "cls_id": self.class_name_map_class_id[cls_name]
        }

        return result

    def __getitem__(self, index):
        data = self.data_all[index]
        result = self._process_image(data)
        return result