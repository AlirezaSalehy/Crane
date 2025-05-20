
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch
import torch.nn as nn

from typing import Union, List
from pkg_resources import packaging

_tokenizer = _Tokenizer()
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        dtype = clip_model.transformer.get_cast_dtype()
        
        self.train_with_img_cls_type = design_details['others'].train_with_img_cls_type
        self.train_with_img_cls_prob = design_details['others'].train_with_img_cls_prob
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        self.n_ctx = design_details["Prompt_length"]
        self.n_ctx_pos = self.n_ctx
        self.n_ctx_neg = self.n_ctx
        
        self.classnames = ["object"]
        self.n_cls = len(self.classnames)
        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]
        self.normal_num = len(self.state_normal_list)
        self.anormaly_num = len(self.state_anomaly_list)

        ###
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            # print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        ###
        ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, self.n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, self.n_ctx_pos, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        # NOTE: removing class description index
        ###
        prompt_prefix_pos = [" ".join(["X"] * self.n_ctx_pos)]
        prompt_prefix_neg = [" ".join(["X"] * self.n_ctx_neg)]  
        classnames = [name.replace("_", " ") for name in self.classnames]
        prompts_pos = [prompt_prefix_pos[idx] +  " " + template.format(name)+ "." for idx, template in enumerate(self.state_normal_list) for name in classnames]
        prompts_neg = [prompt_prefix_neg[idx] +  " " + template.format(name)+ "." for idx, template in enumerate(self.state_anomaly_list) for name in classnames]

        tokenized_prompts_pos = [tokenize(p_pos) for p_pos in prompts_pos]
        tokenized_prompts_neg = [tokenize(p_neg) for p_neg in prompts_neg]
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos) # 'X X X X X X X X X X X X object.'
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg) # 'X X X X X X X X X X X X damaged object.'
        
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            # print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.anormaly_num, self.n_cls, d).permute(1, 0, 2)
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + self.n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + self.n_ctx_neg:, :])
        # print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)
        
        self.constructed_prompts = None
        
    def _pad_and_concatenate_suffix(ctx, selected_embeddings, tokenized_prompts, prefix, suffix, device):
        ctx = torch.cat([ctx.to(device), selected_embeddings.to(device)], dim=3)
        suffix = suffix[:, :, :, :-selected_embeddings.shape[-2], :]
        insert_idx = prefix.shape[-2] + ctx.shape[-2]
        tokenized_prompts = torch.cat(
            [
                tokenized_prompts[:, :, :, :insert_idx].to(device),
                tokenized_prompts[:, :, :, insert_idx].unsqueeze(-1).repeat(1, 1, 1, selected_embeddings.shape[-2]).to(device),
                tokenized_prompts[:, :, :, insert_idx:].to(device)
            ],
            dim=-1
        )
        tokenized_prompts = tokenized_prompts[:, :, :, :-1]
        return ctx, suffix, tokenized_prompts

    def _pad_and_concatenate_prefix(prefix, selected_embeddings, tokenized_prompts, suffix, device):
        prefix = torch.cat([prefix.to(device), selected_embeddings.to(device)], dim=3)
        suffix = suffix[:, :, :, :-selected_embeddings.shape[-2], :]
        insert_idx = prefix.shape[-2]
        tokenized_prompts = torch.cat(
            [
                tokenized_prompts[:, :, :, :insert_idx].to(device),
                tokenized_prompts[:, :, :, insert_idx].unsqueeze(-1).repeat(1, 1, 1, selected_embeddings.shape[-2]).to(device),
                tokenized_prompts[:, :, :, insert_idx:].to(device)
            ],
            dim=-1
        )
        tokenized_prompts = tokenized_prompts[:, :, :, :-1]
        return prefix, suffix, tokenized_prompts

    def _forward(self, img_emb=None):   # Add noise to img_emb   
        if self.train_with_img_cls_prob != 1:
            train_with_img_cls = torch.rand(1).item() <= self.train_with_img_cls_prob
        else:
            train_with_img_cls = True
            
        batch_size = img_emb.shape[0] if not img_emb is None else None
        if train_with_img_cls:   
            assert batch_size == img_emb.shape[0]
                
            # Replicate other tensors if necessary
            prefix_pos = self.token_prefix_pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            suffix_pos = self.token_suffix_pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            prefix_neg = self.token_prefix_neg.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            suffix_neg = self.token_suffix_neg.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            
            # Reshape tokenized prompts to match batch size
            tokenized_prompts_pos = self.tokenized_prompts_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, 1, 77]
            tokenized_prompts_neg = self.tokenized_prompts_neg.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, 1, 77]
            dim = 3
            
            # Replicate ctx_pos and ctx_neg to match batch size
            ctx_pos = self.ctx_pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, 1, 1, 12or6, 768]
            ctx_neg = self.ctx_neg.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, 1, 1, 12or6, 768]

            img_emb = img_emb.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.anormaly_num, 1, 1)

            if self.train_with_img_cls_type == 'replace_prefix':
                ctx_pos = torch.cat([img_emb.to(ctx_pos.device), ctx_pos[:, :, :, 1:, :].to(ctx_pos.device)], dim=3)
                ctx_neg = torch.cat([img_emb.to(ctx_neg.device), ctx_neg[:, :, :, 1:, :].to(ctx_neg.device)], dim=3)
            
            elif self.train_with_img_cls_type == 'replace_suffix':
                ctx_pos = torch.cat([ctx_pos[:, :, :, :-1, :].to(ctx_pos.device), img_emb.to(ctx_pos.device)], dim=3)
                ctx_neg = torch.cat([ctx_neg[:, :, :, :-1, :].to(ctx_neg.device), img_emb.to(ctx_neg.device)], dim=3)
            
            elif self.train_with_img_cls_type == 'pad_prefix':
                prefix_pos, suffix_pos, tokenized_prompts_pos = PromptLearner._pad_and_concatenate_prefix(
                    prefix_pos, img_emb, tokenized_prompts_pos, suffix_pos, prefix_pos.device
                )
                prefix_neg, suffix_neg, tokenized_prompts_neg = PromptLearner._pad_and_concatenate_prefix(
                    prefix_neg, img_emb, tokenized_prompts_neg, suffix_neg, prefix_neg.device
                )
                
            elif self.train_with_img_cls_type == 'pad_suffix':
                ctx_pos, suffix_pos, tokenized_prompts_pos = PromptLearner._pad_and_concatenate_suffix(
                    ctx_pos, img_emb, tokenized_prompts_pos, prefix_pos, suffix_pos, ctx_pos.device
                )
                ctx_neg, suffix_neg, tokenized_prompts_neg = PromptLearner._pad_and_concatenate_suffix(
                    ctx_neg, img_emb, tokenized_prompts_neg, prefix_neg, suffix_neg, ctx_neg.device
                )
            
        else:
            ctx_pos = self.ctx_pos # [1, 1, n_bat, 768]
            ctx_neg = self.ctx_neg
            
            prefix_pos, suffix_pos = self.token_prefix_pos, self.token_suffix_pos
            prefix_neg, suffix_neg = self.token_prefix_neg, self.token_suffix_neg
            tokenized_prompts_pos, tokenized_prompts_neg = self.tokenized_prompts_pos, self.tokenized_prompts_neg
          
            dim = 2
            
        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=dim,
        )
        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=dim,
        )

        l, d = prompts_pos.shape[-2:] # (8, 1, 2, 77, 768)
        prompts_pos = prompts_pos.reshape(-1, l, d)
        l, d = prompts_neg.shape[-2:] # (8, 1, 2, 77, 768)
        prompts_neg = prompts_neg.reshape(-1, l, d)
        
        l, d = tokenized_prompts_pos.shape[-2:]
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(-1, d)
        l, d = tokenized_prompts_neg.shape[-2:]
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(-1, d)
        
        prompts = [prompts_pos, prompts_neg]
        tokenized_prompts = [tokenized_prompts_pos, tokenized_prompts_neg]

        integ_func = torch.stack if train_with_img_cls else torch.cat
        
        prompts = integ_func(prompts, dim=0)
        tokenized_prompts = integ_func(tokenized_prompts, dim = 0)

        return prompts, tokenized_prompts, self.compound_prompts_text, train_with_img_cls
    
    def forward(self, img_emb=None):
        if self.train_with_img_cls_type != 'none' or self.train_with_img_cls_prob != 0 or self.constructed_prompts == None:
            self.constructed_prompts = self._forward(img_emb)
        return self.constructed_prompts