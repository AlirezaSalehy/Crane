import torch
import torch.nn.functional as F  # Import F for direct access to functional operations like interpolate

def calc_similarity_logits(image_features, text_features, temp=0.07): #1/100, 1/25
    image_features_ = image_features.unsqueeze(dim=1) if image_features.dim() == 2 else image_features
    logits = (image_features_ @ text_features.permute(0, 2, 1))/temp
    return logits.squeeze(dim=1) if image_features.dim() == 2 else logits

# mode=nearest (only to check reproducability bcus deterministic)
# relieve is to downsample the groundtruth or use a library which 
# supports autograd   and is deterministic or some trial approach 
# like one dicussed in https://github.com/open-mmlab/mmsegmentation/issues/255 # bilinear
def regrid_upsample(flat_scores, size, mode='bilinear'): 
    h_w = int(flat_scores.shape[1] ** 0.5) 
    regrided = flat_scores.reshape(flat_scores.shape[0], h_w, h_w, -1).permute(0, 3, 1, 2)
    upsampled = F.interpolate(regrided, (size, size), mode=mode).permute(0, 2, 3, 1)
    return upsampled