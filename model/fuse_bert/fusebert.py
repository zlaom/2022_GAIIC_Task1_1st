import torch 
import torch.nn as nn
from model.fuse_bert.fuseembedding import FuseBertEmbeddings
from model.bert.layers import BertEncoder
from typing import List, Optional, Tuple, Union

class FuseBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = FuseBertEmbeddings(config) 
        self.encoder = BertEncoder(config)
        
        # 初始化用的是HERO方法的，huggingface的太复杂看不懂
        self.apply(self.init_weights)

    def forward(self, inputs_embeds, attention_mask=None, token_type_ids=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        # bert
        embedding_output = self.embeddings(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask=extended_attention_mask)

        return encoder_outputs
    
    # 注意只在transformer后面用init_weights，不要包含其他模块
    def init_weights(self, module): 
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask