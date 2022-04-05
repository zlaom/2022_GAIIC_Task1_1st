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
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text = TextModel()
        self.image = nn.Linear(2048, 2048)
    
    @property
    def dtype(self):
        return self.text.transformer.encoder.layer[0].attention.self.key.weight.dtype
    
    def encode_image(self, image):
        # return image.type(self.dtype)
        return self.image(image).type(self.dtype)
    
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