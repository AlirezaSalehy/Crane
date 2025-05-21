from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from utils.transform import *

from segment_anything import sam_model_registry

class SelfCorAttention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, type='qq+kk+vv', qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.type=type

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def _calc_attention_weights(self, k, q, scale=None):
        scale = scale or self.scale
        attn_w = (q @ k.transpose(-2, -1)) * scale
        attn_w = attn_w.softmax(dim=-1)
        attn_w = self.attn_drop(attn_w)
        return attn_w

    def _apply_attention_weights(self, attnw, v, B, N, C):
        x = (attnw @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = self._calc_attention_weights(k, q) # B, num_heads, N, N
        
        if self.type == 'vv':
            attn_s = self._calc_attention_weights(k=v, q=v)
        elif self.type == 'kk':
            attn_s = self._calc_attention_weights(k=k, q=k)
        elif self.type == 'qq':
            attn_s = self._calc_attention_weights(k=q, q=q) 
        elif self.type == 'qq+kk':
            attn_s = self._calc_attention_weights(k=k, q=k) + self._calc_attention_weights(k=q, q=q) 
        elif self.type == 'qq+kk+vv':
            attn_s = self._calc_attention_weights(k=v, q=v) + self._calc_attention_weights(k=k, q=k) + self._calc_attention_weights(k=q, q=q) 

        elif self.type == '(q+k+v)^2':
            scale = self.scale * (3 ** -0.5)
            qkv_concat = torch.cat([q, k, v], dim=-1)
            attn_s = self._calc_attention_weights(k=qkv_concat, q=qkv_concat, scale=scale) 
        else:
            raise ValueError(f"Unsupported attention type: {self.type}. Supported types are: 'vv', 'kk', 'qq', 'qq+kk', 'qq+kk+vv', '(q+k+v)^2'")

        x_ori = self._apply_attention_weights(attn_ori, v, B, N, C)
        x = self._apply_attention_weights(attn_s, v, B, N, C)
        return [x, x_ori]

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, SelfCorAttention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, SelfCorAttention):
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res # skip ffn for the new path NOTE:

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res # NOTE:
                
            return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            
            return x
        
class ResidualAttentionBlock_learnable_token(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
            text_layer=False, i = 0):
        super().__init__(d_model, n_head, attn_mask)
                
        self.i = i
        self.compound_prompt_nctx = design_details['learnabel_text_embedding_length']
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def forward(self, inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            # First check if the ith layer needs compound prompts or not
            if not (counter > len(compound_prompts_deeper) - 1):
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.compound_prompt_nctx:, :, :]
                textual_context = compound_prompts_deeper[counter]
                textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)
                # Once done, update the counter, so that the next time, it does not use same learnable tokens
                counter += 1
                # print(counter)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x[:77], compound_prompts_deeper, counter]

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, design_details = None, text_layer = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_details = design_details
        
        if self.text_layer and (design_details is not None):
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_learnable_token(width, heads, attn_mask, design_details, text_layer, i=i) for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask,) for _ in range(layers)])

    def custom_attn(self, attn_layer, x, dino_feats, beta=1.2, gamma=3, token_size=(16, 16)): 
        if self.design_details['others'].dino_model == 'sam':
            beta=1.08 # To test with SAM although its subpar performance, mean coefficient should be adjusted.
            
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads

        q, k, v = F.linear(x, attn_layer.qkv.weight, attn_layer.qkv.bias).chunk(3, dim=-1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        B, C, H, W = dino_feats.shape
        q_k = F.normalize(dino_feats.flatten(2, 3), dim=1)
        similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)

        similarity = (similarity - torch.mean(similarity) * beta) * gamma
        similarity[similarity < 0] = float('-inf') # NOTE: 

        mask = similarity.to(v.dtype).unsqueeze(1).repeat(1, num_heads, 1, 1)
        mask = mask.reshape(bsz * num_heads, mask.shape[2], mask.shape[3])
        attn_weights = F.softmax(mask, dim=-1)  # Standard attention weights
                    
        v = v[:, 1:, :].reshape(bsz*num_heads, token_size[0], token_size[1], head_dim).permute(0, 3, 1, 2)
        v = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)
        v = v.permute(0, 2, 3, 1).reshape(bsz*num_heads, H*W, head_dim)

        attn_output = torch.bmm(attn_weights, v)  # Apply modified attention weights to V
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.proj(attn_output)

        return attn_output

    def clip_vit_refined_forward(self, x, out_layers, dino_feats=None, feat_idx=1): 
        integrate=self.design_details['others'].both_eattn_dattn
        layers_embds = []
        if not (dino_feats is None):
            for i, r in enumerate(self.resblocks):
                x_out = r(x)
                if i+1 in out_layers:
                    h = w = self.design_details['others'].image_size
                    clip_patch_size = (14, 14)
                    # patch_size = (16, 16) # 
                    token_size = h // clip_patch_size[0], w // clip_patch_size[1]
                
                    layer_input = x[feat_idx] if isinstance(x, list) else x # NOTE: 0 indx self_cor_feats, 1 indx ori_feats
                    dattn_embds =  self.custom_attn(r.attn, r.ln_1(layer_input), dino_feats=dino_feats, token_size=token_size) #
                    
                    layers_embds.append(dattn_embds)
                    if integrate:
                        selfcor_embds = x_out[0] if isinstance(x_out, list) else x_out
                        selfcor_embds = selfcor_embds[1:, :, :]
                        
                        L, B, D = selfcor_embds.shape  #  N, 1369, 768
                        H_W_cur = int(L**0.5)
                        H_W_tgt = int(dattn_embds.shape[0]**0.5)

                        if H_W_tgt != H_W_cur: # interpolate (happens when using dinov1 with higher resolution - patch 8*8)
                            v_rsz = selfcor_embds.permute(1, 0, 2).reshape(B, H_W_cur, H_W_cur, D).permute(0, 3, 1, 2)
                            v_rsz = F.interpolate(v_rsz, size=(H_W_tgt, H_W_tgt), mode='bilinear', align_corners=False)
                            v_rsz = v_rsz.permute(0, 2, 3, 1).reshape(B, (H_W_tgt*H_W_tgt), D).permute(1, 0, 2)
                            layers_embds.append(v_rsz)

                        else:
                            layers_embds.append(selfcor_embds)
                x = x_out

        else:
            for i, r in enumerate(self.resblocks):
                x = r(x)
                if i+1 in out_layers:
                    selfcor_embds = x[0]
                    layers_embds.append(selfcor_embds[1:, :, :] if isinstance(x, list) else x)
        return x, layers_embds
    
    def clip_vit_standard_forward(self, x, out_layers):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[1])
                else:
                    out_tokens.append(x)

        return [x, x], out_tokens

    def forward(self, x: torch.Tensor, out_layers = [6, 12, 18, 24], dino_feats=None, self_cor_attn_layers = None):
        # visual encoder forward
        if not self.text_layer:
            layers_embds = []

            if self_cor_attn_layers is None:
                [x, x], layers_embds = self.clip_vit_standard_forward(x, out_layers)
                return [x, x], layers_embds
            else:
                x, layers_embds = self.clip_vit_refined_forward(x, out_layers, dino_feats=dino_feats)
                return x, layers_embds
            
        # text encoder forward
        else:
            for r in self.resblocks:
                x = r(x)
            return x if self.design_details is None else x[0] # original forward OR learnable text embeddings are inserted

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, design_details=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.no_grad()
    def replace_with_EAttn(self, to_layer, type):
        if to_layer is not None:
            for i in range(1, to_layer):
                self.attn = SelfCorAttention(self.embed_dim, self.embed_dim, self.num_heads, type, True)
                    
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                
                self.transformer.resblocks[-i].attn = self.attn
    
    def _check_interpolate_token_emebeddings(self, new_size):
        n_emb, dim_emb = self.positional_embedding.shape
        ori_size = int((n_emb-1) ** 0.5)

        # update the position embedding during inference for varied input size, position embeddings are flat
        if new_size != ori_size:
            new_pos = self.positional_embedding[1:, :].reshape(-1, ori_size, ori_size, dim_emb).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_size, new_size), mode='bilinear')
            new_pos = new_pos.reshape(-1, dim_emb, new_size * new_size).transpose(1, 2)        
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, features_list, dino_feats=None, self_cor_attn_layers = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        new_size = int((x.shape[1] - 1) ** 0.5)
        self._check_interpolate_token_emebeddings(new_size)

        pos = self.positional_embedding.to(x.dtype)
        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        [x, x_ori], patch_tokens = self.transformer(x, features_list, dino_feats=dino_feats, self_cor_attn_layers = self_cor_attn_layers)
        
        patch_token_list = []
        for patch_token in patch_tokens:
            patch_token = self.ln_post(patch_token.permute(1, 0, 2)) @ self.proj  # LND -> NLD
            patch_token_list.append(patch_token)
        patch_tokens = patch_token_list

        x_ori = x_ori.permute(1, 0, 2)
        x_ori = self.ln_post(x_ori)
        x_ori = x_ori @ self.proj
        return x_ori[:, 0, :], patch_tokens

class ScoreBasePooling(nn.Module):
    def __init__(self):
        super(ScoreBasePooling, self).__init__()

    def prepare_anomamps(self, anomaly_maps, softmax_first):
        anomaly_map = torch.stack(anomaly_maps, dim=1)
        
        if softmax_first:
            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, :, 1].unsqueeze(dim=-1) # B, L
            anomaly_map = torch.mean(anomaly_map, dim=1)
        else:
            anomaly_map = torch.mean(anomaly_map, dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1].unsqueeze(dim=-1) # B, L
            
        return anomaly_map
    
    def select_weights(self, anomaly_map, threshold=-1, top_k=-1):
        if threshold > 0:
            mask = anomaly_map > threshold
            anomaly_map = anomaly_map * mask.float()
        elif top_k > 0:
            N, H, W = anomaly_map.shape
            flattened_map = anomaly_map.view(N, -1)  # Shape: (N, H*W)
            _, top_k_indices = torch.topk(flattened_map, k=top_k, dim=-1)
            mask = torch.zeros_like(flattened_map, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
            mask = mask.view(N, H, W)
            anomaly_map = anomaly_map * mask.float()
            
        return anomaly_map
        
    def forward(self, patch_tokens: list, anomaly_maps: list): # 4, N, L, C
        patch_tokens = torch.mean(patch_tokens, dim=0)
        anomaly_map = self.prepare_anomamps(anomaly_maps, softmax_first=False) # N, L, 1
        img_cls = torch.sum(patch_tokens*anomaly_map, dim=1)
        img_cls = F.normalize(img_cls, dim=1)
        return img_cls

class Crane(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details = None
                 ):
        super().__init__()

        self.context_length = context_length
            
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            design_details=design_details
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), text_layer=True, design_details=design_details
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.adapter_image = None 
        self.vfm = None
        self.image_resolution = design_details['others'].image_size
        
        self.initialize_parameters()
    
    def use_DAttn(self, dino):
        self.vfm_model = dino 
        
        if self.vfm_model == 'dino':
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm_patch_size = 8
            
        elif self.vfm_model == 'dinov2':
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            self.vfm_patch_size = 14
            
        elif self.vfm_model == 'sam': # SAM-ViT-B	
            self.vfm = sam_model_registry["vit_b"](checkpoint='~/.cache/sam/sam_vit_b_01ec64.pth')
            self.vfm_patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size

        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list = [], self_cor_attn_layers = None):
        dino_feats = None
        if self.vfm:
            raw_img = unnormalize(image, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
            dino_img = Normalize(IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD)(raw_img)

            if self.vfm_model == 'dino':
                feat = self.vfm.get_intermediate_layers(dino_img)[0]
                t_l = self.image_resolution//self.vfm_patch_size 
                dino_feats = feat[:, 1:, :].reshape(-1, t_l, t_l, feat.shape[2]).permute(0, 3, 1, 2)
            
            elif self.vfm_model == 'dinov2':
                dino_feats = self.vfm.get_intermediate_layers(dino_img, reshape=True)[0]

            elif self.vfm_model == 'sam':
                patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size
                imgs_norm = F.interpolate(dino_img, size=(1024, 1024), mode='bilinear', align_corners=False)
                I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
                dino_feats = self.vfm.image_encoder(imgs_norm) # [B, 256, 64, 64]

        image_features, patch_features = self.visual(image.type(self.dtype), feature_list, dino_feats=dino_feats, self_cor_attn_layers=self_cor_attn_layers)
        return image_features, patch_features

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND 12, 77, 768
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text = None):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts + self.positional_embedding.to(cast_dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
