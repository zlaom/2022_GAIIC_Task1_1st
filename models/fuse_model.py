import torch
import torch.nn as nn
import collections
from models.split_bert.split_bert import SplitBertModel
from models.fuse_bert.fuse_bert import FuseBertModel
from models.hero_bert.tokenizer import Tokenizer
from einops import rearrange, repeat

class ImageModel(nn.Module):
    def __init__(self, img_dim=2048, hidden_size=768, p=0.2, n_img_expand=6):
        super().__init__()
        self.n_img_expand = n_img_expand
        self.image_encoder = nn.ModuleList()
        for i in range(n_img_expand):
            self.image_encoder.append(
                nn.Sequential(nn.Dropout(p=p),
                nn.Linear(img_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                )
            )
        
    def forward(self, image):
        outputs = []
        for i in range(self.n_img_expand):
            hidden = self.image_encoder[i](image)
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=1)
        return outputs
        

class FuseModel(nn.Module):
    def __init__(self, split_config, fuse_config, vocab_file,img_p=0.2, img_dim=2048, n_img_expand=8):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.fusebert = FuseBertModel(fuse_config)
        self.tokenizer = Tokenizer(vocab_file)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        
        self.image_encoder = ImageModel(img_dim=img_dim, hidden_size=fuse_config.hidden_size, p=0.2, n_img_expand=n_img_expand)
        
        self.head = nn.Linear(fuse_config.hidden_size * 2 , 2)
        # self.head = nn.Linear(fuse_config.hidden_size*2, 2)
        

    def forward(self, image_features, splits): # 注意splits需要为二维列表
        B = image_features.shape[0]
        image_features = self.image_encoder(image_features)
        
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
        cls_embedding = fuse_output[:, 0, :]
        mean_embedding = torch.mean(fuse_output[:, 1:, :], dim=1)
        final_embdding = torch.cat([cls_embedding,  mean_embedding], dim=-1)
        x = self.head(final_embdding)
        return x


