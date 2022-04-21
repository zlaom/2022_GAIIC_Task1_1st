import torch
import torch.nn as nn
from model.split_bert.splitbert import SplitBertModel
from model.fuse_bert.fusebert import FuseBertModel
from model.bert.tokenizer import Tokenizer

from einops import rearrange, repeat

class FuseModel(nn.Module):
    def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.fusebert = FuseBertModel(fuse_config)
        self.tokenizer = Tokenizer(vocab_file)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
            nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
        )
        
        self.head = nn.Linear(fuse_config.hidden_size, 1)
           
    def forward(self, image_features, splits, word_match=False): # 注意splits需要为二维列表
        B = image_features.shape[0]
        image_features = self.image_encoder(image_features)
        image_features = image_features.reshape(B, self.n_img_expand, -1)
        
        # 构建split输入
        tokens = self.tokenizer(splits)
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        
        # split bert
        sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        split_seq_len = sequence_output.shape[1]
        
        # 构建fuse输入
        cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
        fuse_inputs_embeds = torch.cat([cls_tokens, image_features, sequence_output], dim=1)
        fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
                                         tokens['attention_mask']], dim=1).cuda()
        fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
                                         torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
        
        # fuse bert
        fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
                                    attention_mask=fuse_attention_mask,
                                    token_type_ids=fuse_token_type_ids)[0]
        
        # 输出头
        if word_match:
            x = self.head(fuse_output)[:,self.n_img_expand+1:,:]
            return x, attention_mask
        
        x = self.head(fuse_output[:,0,:])
        return x



class FuseModelWithFusehead(nn.Module):
    def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.fusebert = FuseBertModel(fuse_config)
        self.tokenizer = Tokenizer(vocab_file)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
            nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
        )
        
        self.head = nn.Linear(fuse_config.hidden_size, 1)
        self.proj = nn.Sequential(nn.Linear(fuse_config.hidden_size, fuse_config.hidden_size),
                                  nn.LayerNorm(fuse_config.hidden_size),
                                  nn.ReLU(inplace=True))
        self.fuse_head = nn.Linear(fuse_config.hidden_size + fuse_config.hidden_size, 1)
        

    def forward(self, image_features, splits, word_match=False): # 注意splits需要为二维列表
        B = image_features.shape[0]
        image_features = self.image_encoder(image_features)
        image_features = image_features.reshape(B, self.n_img_expand, -1)
        
        # 构建split输入
        tokens = self.tokenizer(splits)
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        
        # split bert
        sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        split_seq_len = sequence_output.shape[1]
        
        # 构建fuse输入
        cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
        fuse_inputs_embeds = torch.cat([cls_tokens, image_features, sequence_output], dim=1)
        fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
                                         tokens['attention_mask']], dim=1).cuda()
        fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
                                         torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
        
        # fuse bert
        fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
                                    attention_mask=fuse_attention_mask,
                                    token_type_ids=fuse_token_type_ids)[0]
        
        # 输出头
        if word_match:
            title_output = fuse_output[:,self.n_img_expand+1:,:]
            B, W, D = title_output.shape
            cls_token = fuse_output[:, 0, :] # 这步数据会变成2维
            cls_token = self.proj(cls_token)
            cls_token_expand = cls_token.unsqueeze(1).expand(B, W, D)
            final_outputs = torch.cat([title_output, cls_token_expand], dim=-1)
            x = self.fuse_head(final_outputs)
            return x, attention_mask
        
        x = self.head(fuse_output[:,0,:])
        return x
