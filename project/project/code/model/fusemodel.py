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
        self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
        self.image_encoder = nn.Sequential(
            nn.Dropout(fuse_config.image_dropout),
            nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
            nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
        )
        
        self.head = nn.Linear(fuse_config.hidden_size, 1)

    def forward(self, image_features, splits): # 注意splits需要为二维列表
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
        # position embedding
        pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
        pos_embeded += self.pos_embedding
        fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
        fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
                                         tokens['attention_mask']], dim=1).cuda()
        # fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
        #                                  torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
        fuse_token_type_ids = None 
        
        # fuse bert
        fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
                                    attention_mask=fuse_attention_mask,
                                    token_type_ids=fuse_token_type_ids)[0]
        
        # 输出头
        x = self.head(fuse_output[:,0,:])
        return x



class DesignFuseModel(nn.Module):
    def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8, word_match=False):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.fusebert = FuseBertModel(fuse_config)
        self.tokenizer = Tokenizer(vocab_file)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
            nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
        )
        self.dropout = nn.Dropout(fuse_config.image_dropout)
        self.head = nn.Linear(fuse_config.hidden_size, 1)
        
        self.word_match = word_match
        if self.word_match:
            self.word_head = nn.Linear(fuse_config.hidden_size, 1)
        
    def forward(self, image_features, splits): # 注意splits需要为二维列表
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
        # position embedding
        pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
        pos_embeded += self.pos_embedding
        pos_embeded = self.dropout(pos_embeded)
        fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
        fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
                                         tokens['attention_mask']], dim=1).cuda()
        fuse_token_type_ids = None 
        
        # fuse bert
        fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
                                    attention_mask=fuse_attention_mask,
                                    token_type_ids=fuse_token_type_ids)[0]
        
        # 输出头
        if self.word_match:
            title_match = self.head(fuse_output[:,0,:]).squeeze(1)
            word_match = self.word_head(fuse_output)[:,self.n_img_expand+1:,:].squeeze(2)
            return title_match, word_match, attention_mask
        
        title_match = self.head(fuse_output[:,0,:]).squeeze(1)
        return title_match




class DesignFuseModelMean(nn.Module):
    def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8, word_match=False):
        super().__init__()
        self.n_img_expand = n_img_expand
        
        self.splitbert = SplitBertModel(split_config)
        self.fusebert = FuseBertModel(fuse_config)
        self.tokenizer = Tokenizer(vocab_file)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
            nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
        )
        self.dropout = nn.Dropout(fuse_config.image_dropout)
        self.head = nn.Linear(fuse_config.hidden_size, 1)
        
        self.word_match = word_match
        if self.word_match:
            self.word_head = nn.Linear(fuse_config.hidden_size, 1)
        
    def forward(self, image_features, splits): # 注意splits需要为二维列表
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
        # position embedding
        pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
        pos_embeded += self.pos_embedding
        pos_embeded = self.dropout(pos_embeded)
        fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
        fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
                                         tokens['attention_mask']], dim=1).cuda()
        fuse_token_type_ids = None 
        
        # fuse bert
        fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
                                    attention_mask=fuse_attention_mask,
                                    token_type_ids=fuse_token_type_ids)[0]
        
        # 输出头
        if self.word_match:
            word_match = self.word_head(fuse_output[:,self.n_img_expand+1:,:]).squeeze(2)

            mean = torch.mean(fuse_output, dim=1)
            title_match = self.head(mean).squeeze(1)
            return title_match, word_match, attention_mask
        
        title_match = self.head(fuse_output[:,0,:]).squeeze(1)
        return title_match



# class ImageModel(nn.Module):
#     def __init__(self, img_dim=2048, hidden_size=768, p=0.2, n_img_expand=6):
#         super().__init__()
#         self.n_img_expand = n_img_expand
#         self.image_encoder = nn.ModuleList()
#         for i in range(n_img_expand):
#             self.image_encoder.append(
#                 nn.Sequential(nn.Dropout(p=p),
#                 nn.Linear(img_dim, hidden_size),
#                 nn.LayerNorm(hidden_size),
#                 )
#             )
        
#     def forward(self, image):
#         outputs = []
#         for i in range(self.n_img_expand):
#             hidden = self.image_encoder[i](image)
#             outputs.append(hidden)
#         outputs = torch.stack(outputs, dim=1)
#         return outputs


# class FuseModel2TasksNewDropout(nn.Module):
#     def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8, word_match=False):
#         super().__init__()
#         self.n_img_expand = n_img_expand
        
#         self.splitbert = SplitBertModel(split_config)
#         self.fusebert = FuseBertModel(fuse_config)
#         self.tokenizer = Tokenizer(vocab_file)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
#         self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
#         self.image_encoder = ImageModel(img_dim=img_dim, 
#                                         hidden_size=fuse_config.hidden_size, 
#                                         p=fuse_config.image_dropout, 
#                                         n_img_expand=n_img_expand)
#         self.dropout = nn.Dropout(fuse_config.image_dropout)
#         self.head = nn.Linear(fuse_config.hidden_size, 1)
        
#         self.word_match = word_match
#         if self.word_match:
#             self.word_head = nn.Linear(fuse_config.hidden_size, 1)
        
#     def forward(self, image_features, splits): # 注意splits需要为二维列表
#         B = image_features.shape[0]
#         image_features = self.image_encoder(image_features)
#         # image_features = image_features.reshape(B, self.n_img_expand, -1)
        
#         # 构建split输入
#         tokens = self.tokenizer(splits)
#         input_ids = tokens['input_ids'].cuda()
#         attention_mask = tokens['attention_mask'].cuda()
        
#         # split bert
#         sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         split_seq_len = sequence_output.shape[1]
        
#         # 构建fuse输入
#         cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
#         # position embedding
#         pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
#         pos_embeded += self.pos_embedding
#         pos_embeded = self.dropout(pos_embeded)
#         fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
#         fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
#                                          tokens['attention_mask']], dim=1).cuda()
#         fuse_token_type_ids = None 
        
#         # fuse bert
#         fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
#                                     attention_mask=fuse_attention_mask,
#                                     token_type_ids=fuse_token_type_ids)[0]
        
#         # 输出头
#         if self.word_match:
#             title_match = self.head(fuse_output[:,0,:]).squeeze(1)
#             word_match = self.word_head(fuse_output)[:,self.n_img_expand+1:,:].squeeze(2)
#             return title_match, word_match, attention_mask
        
#         title_match = self.head(fuse_output[:,0,:]).squeeze(1)
#         return title_match





# class DesignFuseModelMean(nn.Module):
#     def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8, word_match=False):
#         super().__init__()
#         self.n_img_expand = n_img_expand
        
#         self.splitbert = SplitBertModel(split_config)
#         self.fusebert = FuseBertModel(fuse_config)
#         self.tokenizer = Tokenizer(vocab_file)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
#         self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
#         self.image_encoder = nn.Sequential(
#             nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
#             nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
#         )
#         self.dropout = nn.Dropout(fuse_config.image_dropout)
#         self.head = nn.Linear(fuse_config.hidden_size * 2, 1)
        
#         self.word_match = word_match
#         if self.word_match:
#             self.word_head = nn.Linear(fuse_config.hidden_size, 1)
        
#     def forward(self, image_features, splits): # 注意splits需要为二维列表
#         B = image_features.shape[0]
#         image_features = self.image_encoder(image_features)
#         image_features = image_features.reshape(B, self.n_img_expand, -1)
        
#         # 构建split输入
#         tokens = self.tokenizer(splits)
#         input_ids = tokens['input_ids'].cuda()
#         attention_mask = tokens['attention_mask'].cuda()
        
#         # split bert
#         sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         split_seq_len = sequence_output.shape[1]
        
#         # 构建fuse输入
#         cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
#         # position embedding
#         pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
#         pos_embeded += self.pos_embedding
#         pos_embeded = self.dropout(pos_embeded)
#         fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
#         fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
#                                          tokens['attention_mask']], dim=1).cuda()
#         fuse_token_type_ids = None 
        
#         # fuse bert
#         fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
#                                     attention_mask=fuse_attention_mask,
#                                     token_type_ids=fuse_token_type_ids)[0]
        
#         # 输出头
#         if self.word_match:
#             word_match = self.word_head(fuse_output)[:,self.n_img_expand+1:,:].squeeze(2)
#             meanpool = torch.mean(fuse_output, dim=1)
#             title_match = torch.cat([fuse_output[:,0,:], meanpool], dim=1)
#             title_match = self.head(title_match).squeeze(1)
#             return title_match, word_match, attention_mask
        
#         title_match = self.head(fuse_output[:,0,:]).squeeze(1)
#         return title_match










# class FuseModelToken(nn.Module):
#     def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8):
#         super().__init__()
#         self.n_img_expand = n_img_expand
        
#         self.splitbert = SplitBertModel(split_config)
#         self.fusebert = FuseBertModel(fuse_config)
#         self.tokenizer = Tokenizer(vocab_file)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
#         # self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
#         self.image_encoder = nn.Sequential(
#             # nn.Dropout(0.3),
#             nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
#             nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
#         )
        
#         self.head = nn.Linear(fuse_config.hidden_size, 1)
           
#     def forward(self, image_features, splits, word_match=False): # 注意splits需要为二维列表
#         B = image_features.shape[0]
#         image_features = self.image_encoder(image_features)
#         image_features = image_features.reshape(B, self.n_img_expand, -1)
        
#         # 构建split输入
#         tokens = self.tokenizer(splits)
#         input_ids = tokens['input_ids'].cuda()
#         attention_mask = tokens['attention_mask'].cuda()
        
#         # split bert
#         sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         split_seq_len = sequence_output.shape[1]
        
#         # 构建fuse输入
#         cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
#         # position embedding
#         pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
#         # pos_embeded += self.pos_embedding
#         fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
#         fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
#                                          tokens['attention_mask']], dim=1).cuda()
#         fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
#                                          torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
#         # fuse_token_type_ids = None 
        
#         # fuse bert
#         fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
#                                     attention_mask=fuse_attention_mask,
#                                     token_type_ids=fuse_token_type_ids)[0]
        
#         # 输出头
#         if word_match:
#             x = self.head(fuse_output)[:,self.n_img_expand+1:,:]
#             return x, attention_mask
        
#         x = self.head(fuse_output[:,0,:])
#         return x

# class FuseModelCE(nn.Module):
#     def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8):
#         super().__init__()
#         self.n_img_expand = n_img_expand
        
#         self.splitbert = SplitBertModel(split_config)
#         self.fusebert = FuseBertModel(fuse_config)
#         self.tokenizer = Tokenizer(vocab_file)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
#         self.pos_embedding = nn.Parameter(torch.randn(1, n_img_expand + 1, fuse_config.hidden_size))
        
#         self.image_encoder = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
#             nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
#         )
        
#         self.head = nn.Linear(fuse_config.hidden_size, 2)
           
#     def forward(self, image_features, splits, word_match=False): # 注意splits需要为二维列表
#         B = image_features.shape[0]
#         image_features = self.image_encoder(image_features)
#         image_features = image_features.reshape(B, self.n_img_expand, -1)
        
#         # 构建split输入
#         tokens = self.tokenizer(splits)
#         input_ids = tokens['input_ids'].cuda()
#         attention_mask = tokens['attention_mask'].cuda()
        
#         # split bert
#         sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         split_seq_len = sequence_output.shape[1]
        
#         # 构建fuse输入
#         cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
#         # position embedding
#         pos_embeded = torch.cat([cls_tokens, image_features], dim=1)
#         pos_embeded += self.pos_embedding
#         fuse_inputs_embeds = torch.cat([pos_embeded, sequence_output], dim=1)
#         fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
#                                          tokens['attention_mask']], dim=1).cuda()
#         # fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
#         #                                  torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
#         fuse_token_type_ids = None 
        
#         # fuse bert
#         fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
#                                     attention_mask=fuse_attention_mask,
#                                     token_type_ids=fuse_token_type_ids)[0]
        
#         # 输出头
#         if word_match:
#             x = self.head(fuse_output)[:,self.n_img_expand+1:,:]
#             return x, attention_mask
        
#         x = self.head(fuse_output[:,0,:])
#         return x
    
    
    

# class OneWordFuseModel(nn.Module):
#     def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8):
#         super().__init__()
#         self.n_img_expand = n_img_expand
        
#         self.splitbert = SplitBertModel(split_config)
#         self.fusebert = FuseBertModel(fuse_config)
#         self.tokenizer = Tokenizer(vocab_file)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        
#         self.image_encoder = nn.Sequential(
#             nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
#             nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
#         )
        
#         self.head = nn.Linear(fuse_config.hidden_size, 1)
           
#     def forward(self, image_features, splits, word_match=False): # 注意splits需要为二维列表
#         B = image_features.shape[0]
#         image_features = self.image_encoder(image_features)
#         image_features = image_features.reshape(B, self.n_img_expand, -1)
        
#         # 构建split输入
#         tokens = self.tokenizer(splits)
#         input_ids = tokens['input_ids'].cuda()
#         attention_mask = tokens['attention_mask'].cuda()
        
#         # split bert
#         sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         split_seq_len = sequence_output.shape[1]
        
#         # 构建fuse输入
#         cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
#         fuse_inputs_embeds = torch.cat([cls_tokens, image_features, sequence_output], dim=1)
#         fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
#                                          tokens['attention_mask']], dim=1).cuda()
#         fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
#                                          torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
        
#         # fuse bert
#         fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
#                                     attention_mask=fuse_attention_mask,
#                                     token_type_ids=fuse_token_type_ids)[0]
        
#         # 输出头
#         if word_match:
#             x = self.head(fuse_output)[:,self.n_img_expand+1:,:]
#             return x, attention_mask
        
#         x = self.head(fuse_output[:,0,:])
#         return x
    
    
    
# class FuseModelWithFusehead(nn.Module):
#     def __init__(self, split_config, fuse_config, vocab_file, img_dim=2048, n_img_expand=8):
#         super().__init__()
#         self.n_img_expand = n_img_expand
        
#         self.splitbert = SplitBertModel(split_config)
#         self.fusebert = FuseBertModel(fuse_config)
#         self.tokenizer = Tokenizer(vocab_file)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, fuse_config.hidden_size))
        
#         self.image_encoder = nn.Sequential(
#             nn.Linear(img_dim, fuse_config.hidden_size * n_img_expand),
#             nn.LayerNorm(fuse_config.hidden_size * n_img_expand)
#         )
        
#         self.head = nn.Linear(fuse_config.hidden_size, 1)
#         self.proj = nn.Sequential(nn.Linear(fuse_config.hidden_size, fuse_config.hidden_size),
#                                   nn.LayerNorm(fuse_config.hidden_size),
#                                   nn.ReLU(inplace=True))
#         self.fuse_head = nn.Linear(fuse_config.hidden_size + fuse_config.hidden_size, 1)
        

#     def forward(self, image_features, splits, word_match=False): # 注意splits需要为二维列表
#         B = image_features.shape[0]
#         image_features = self.image_encoder(image_features)
#         image_features = image_features.reshape(B, self.n_img_expand, -1)
        
#         # 构建split输入
#         tokens = self.tokenizer(splits)
#         input_ids = tokens['input_ids'].cuda()
#         attention_mask = tokens['attention_mask'].cuda()
        
#         # split bert
#         sequence_output = self.splitbert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         split_seq_len = sequence_output.shape[1]
        
#         # 构建fuse输入
#         cls_tokens = repeat(self.cls_token, '1 N D -> B N D', B = B).cuda()
#         fuse_inputs_embeds = torch.cat([cls_tokens, image_features, sequence_output], dim=1)
#         fuse_attention_mask = torch.cat([torch.ones(B, self.n_img_expand+1, dtype=int), 
#                                          tokens['attention_mask']], dim=1).cuda()
#         fuse_token_type_ids = torch.cat([torch.zeros(B, self.n_img_expand+1, dtype=int),
#                                          torch.ones(B, split_seq_len, dtype=int)], dim=1).cuda()
        
#         # fuse bert
#         fuse_output = self.fusebert(inputs_embeds=fuse_inputs_embeds, 
#                                     attention_mask=fuse_attention_mask,
#                                     token_type_ids=fuse_token_type_ids)[0]
        
#         # 输出头
#         if word_match:
#             title_output = fuse_output[:,self.n_img_expand+1:,:]
#             B, W, D = title_output.shape
#             cls_token = fuse_output[:, 0, :] # 这步数据会变成2维
#             cls_token = self.proj(cls_token)
#             cls_token_expand = cls_token.unsqueeze(1).expand(B, W, D)
#             final_outputs = torch.cat([title_output, cls_token_expand], dim=-1)
#             x = self.fuse_head(final_outputs)
#             return x, attention_mask
        
#         x = self.head(fuse_output[:,0,:])
#         return x
