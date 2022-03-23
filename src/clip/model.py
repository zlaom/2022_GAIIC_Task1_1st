from collections import OrderedDict
from typing import Tuple, Union

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import transformers


class TextModel(torch.nn.Module):
    def __init__(self, model_name='M-CLIP/M-BERT-Base-69', out_features=2048):
        super().__init__()
        self.model_name = model_name
        self.transformer = transformers.AutoModel.from_pretrained(model_name, cache_dir='~/.cache')
        in_features = self.transformer.pooler.dense.out_features
        self.clip_head = torch.nn.Linear(in_features=in_features, out_features=out_features)
    
    def forward(self, txt_tok):
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.clip_head(embs)


class CLIP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text = TextModel()
    
    @property
    def dtype(self):
        return self.text.transformer.encoder.layer[0].attention.self.key.weight.dtype
    
    def encode_image(self, image):
        return image.type(self.dtype)
    
    def encode_text(self, text):
        return self.text(text).type(self.dtype)
    
    def forward(self, image, text):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features, self.logit_scale.exp()
    
    
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
    
    
def build_model():
    model = CLIP()
    convert_weights(model)
    return model.eval()
