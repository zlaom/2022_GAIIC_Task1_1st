import torch 
import torch.nn as nn
from model.cross_bert.crosslayers import CrossBertEncoder

class CrossBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = CrossBertEncoder(config)
        
        # 初始化用的是HERO方法的，huggingface的太复杂看不懂
        self.apply(self.init_weights)

    def forward(self, inputs_embeds1, inputs_embeds2, attention_mask1=None, attention_mask2=None):
        extended_attention_mask1 = self.get_extended_attention_mask(attention_mask1)
        extended_attention_mask2 = self.get_extended_attention_mask(attention_mask2)
        # crossbert
        encoder_outputs1, encoder_outputs2 = self.encoder(inputs_embeds1, inputs_embeds2, extended_attention_mask1, extended_attention_mask2)

        return encoder_outputs1, encoder_outputs2
    
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