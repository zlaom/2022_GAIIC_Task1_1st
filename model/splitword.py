import torch 
import torch.nn as nn
from model.bert.embedding import BertEmbeddings
from model.bert.layers import BertEncoder
from typing import List, Optional, Tuple, Union

class SplitWordModel(nn.Module):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        # 初始化用的是HERO方法的，huggingface的太复杂看不懂
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # input_shape = input_ids.size()
        # batch_size, seq_length = input_shape
        # device = input_ids.device
        
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(input_ids=input_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]

        return sequence_output

    def init_weights(self, module): # 注意只在transformer后面用，不要包含其他模块
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