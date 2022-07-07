import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None):
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# class FuseBertEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings."""

#     def __init__(self, config):
#         super().__init__()
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, inputs_embeds=None, token_type_ids=None):
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#         embeddings = inputs_embeds + token_type_embeddings
        
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings
    
    
# huggingface embedding类的原型，上面是自己的简化版
# class BertPosEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings."""

#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
#         # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#         # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
#         # self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

#     def forward(
#         self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
#     ):
#         # if input_ids is not None:
#         #     input_shape = input_ids.size()
#         # else:
#         #     input_shape = inputs_embeds.size()[:-1]

#         # seq_length = input_shape[1]

#         # if position_ids is None:
#         #     position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)

#         embeddings = inputs_embeds
        
#         # if self.position_embedding_type == "absolute":
#         #     position_embeddings = self.position_embeddings(position_ids)
#         #     embeddings += position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings