from msilib import sequence
import torch
import torch.nn as nn
from model.split_bert.splitbert import SplitBertModel
from model.cross_bert.crossbert import CrossBert
from model.bert.tokenizer import Tokenizer

from einops import rearrange, repeat

class PretrainCrossModel(nn.Module):
    def __init__(self, split_config, cross_config, vocab_file, img_dim=2048, n_img_expand=8):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.crossbert = CrossBert(cross_config)
        self.tokenizer = Tokenizer(vocab_file)
        
        
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, cross_config.hidden_size * n_img_expand),
            nn.LayerNorm(cross_config.hidden_size * n_img_expand)
        )
        
        # 预训练分类头
        self.word_head = nn.Linear(cross_config.hidden_size, 1)
        
    def forward(self, image_features, splits): # 注意splits需要为二维列表
        B = image_features.shape[0]
        image_features = self.image_encoder(image_features)
        image_embeds = image_features.reshape(B, self.n_img_expand, -1)
        
        # 构建split输入
        tokens = self.tokenizer(splits)
        input_ids = tokens['input_ids'].cuda()
        split_attention_mask = tokens['attention_mask'].cuda()
        # split bert
        split_embeds = self.splitbert(input_ids=input_ids, attention_mask=split_attention_mask)[0]
        
        # 构建cross输入
        image_attention_mask = torch.ones(B, self.n_img_expand, dtype=int).cuda()
        # cross bert
        image_outputs, split_outputs = self.fusebert(image_embeds, split_embeds, 
                                                     image_attention_mask, split_attention_mask)
        image_embeds =  image_outputs[0]
        split_embeds = split_outputs[0]

        # 输出头
        x = self.word_head(split_embeds)[:,self.n_img_expand:,:]
        return x, split_attention_mask

    
    
class FinetuneCrossModel(nn.Module):
    def __init__(self, split_config, cross_config, cls_split_config, vocab_file, img_dim=2048, n_img_expand=8):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.crossbert = CrossBert(cross_config)
        self.tokenizer = Tokenizer(vocab_file)
        
        
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, cross_config.hidden_size * n_img_expand),
            nn.LayerNorm(cross_config.hidden_size * n_img_expand)
        )
        
        # cls_token和分类头以及第二个splitbert是finetune才用到的
        self.cls_splitbert = SplitBertModel(cls_split_config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cls_split_config.hidden_size))
        self.cls_head = nn.Linear(cls_split_config.hidden_size, 1)
           
        

    def forward(self, image_features, splits): # 注意splits需要为二维列表
        B = image_features.shape[0]
        image_features = self.image_encoder(image_features)
        image_embeds = image_features.reshape(B, self.n_img_expand, -1)
        
        # 构建split输入
        tokens = self.tokenizer(splits)
        input_ids = tokens['input_ids'].cuda()
        split_attention_mask = tokens['attention_mask'].cuda()
        # split bert
        split_embeds = self.splitbert(input_ids=input_ids, attention_mask=split_attention_mask)[0]
        
        # 构建cross输入
        image_attention_mask = torch.ones(B, self.n_img_expand, dtype=int).cuda()
        # cross bert
        image_outputs, split_outputs = self.fusebert(image_embeds, split_embeds, 
                                                     image_attention_mask, split_attention_mask)
        image_embeds =  image_outputs[0]
        split_embeds = split_outputs[0]
        
        # 构建cls输入
        cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
        cls_embeds = torch.cat([cls_tokens, split_embeds], dim=1)
        cls_attention_mask = torch.cat([torch.ones(B, 1, dtype=int), split_attention_mask], dim=1)
        # cls split bert
        cls_embeds = self.cls_splitbert(input_embeds=cls_embeds, 
                                        attention_mask=cls_attention_mask).cuda()
        
        # 输出头
        x = self.cls_head(cls_embeds[:,0,:])
        return x
