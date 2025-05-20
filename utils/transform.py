from typing import Optional, Tuple
import torchvision.transforms as transforms
import torch

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    unnormalized_tensor = tensor * std + mean
    return unnormalized_tensor.clamp(0, 1)

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(image_size, mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    tnsfrms = [
        transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(image_size), # NOTE: No need for centercrop
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(tnsfrms)

def get_transform(args):
    input_transform = image_transform(args.image_size, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    # input_transform.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None) # NOTE: put antialias
    # input_transform.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size)) 
    
    label_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.CenterCrop(args.image_size), # NOTE: No need for centercrop
        transforms.ToTensor()
    ])

    return input_transform, label_transform
