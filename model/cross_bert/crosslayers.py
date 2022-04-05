"""
BERT/RoBERTa layers from the huggingface implementation
(https://github.com/huggingface/transformers)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm as BertLayerNorm

import math


def gelu(x):
    """ Original Implementation of the gelu activation function
        in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
            * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently
        in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (
        1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish, "gelu_new": gelu_new}


class MLPLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz):
        super(MLPLayer, self).__init__()
        self.linear_1 = nn.Linear(in_hsz, in_hsz*2)
        self.LayerNorm = BertLayerNorm(in_hsz*2, eps=1e-5)
        self.linear_2 = nn.Linear(in_hsz*2, out_hsz)
        self.act = gelu

    def forward(self, x):
        x_1 = self.linear_1(x)
        x_1 = self.act(x_1)
        x_1 = self.LayerNorm(x_1)
        x_2 = self.linear_2(x_1)
        return x_2


class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True,
                 dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = BertLayerNorm(in_hsz, eps=1e-5)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        # hidden维度需要是head的整数倍
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of "
                            "the number of attention heads (%d)" % (
                            config.hidden_size, config.num_attention_heads))
            
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.hidden_size, self.all_head_size)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # [B,W,D]->[B,W,H,Dh]->[B,H,W,Dh]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def get_outputs(self, mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask=None):
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # 恢复原来的维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs 
    
    def forward(self, hidden_states1, hidden_states2, attention_mask1=None, attention_mask2=None):
        # get qkv1
        mixed_query_layer1 = self.query1(hidden_states1) # [B,W,D]->[B,W,D]
        mixed_key_layer1 = self.key1(hidden_states1)
        mixed_value_layer1 = self.value1(hidden_states1)

        # get qkv2
        mixed_query_layer2 = self.query2(hidden_states2) # [B,W,D]->[B,W,D]
        mixed_key_layer2 = self.key2(hidden_states2)
        mixed_value_layer2 = self.value2(hidden_states2)

        outputs1 = self.get_outputs(mixed_query_layer1, mixed_key_layer2, mixed_value_layer2, attention_mask2)
        outputs2 = self.get_outputs(mixed_query_layer2, mixed_key_layer1, mixed_value_layer1, attention_mask1)
        return outputs1, outputs2


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.cross = BertCrossAttention(config)
        self.output1 = BertSelfOutput(config)
        self.output2 = BertSelfOutput(config)
    
    # 传入input_tensor用来做残差
    def forward(self, input_tensor1, input_tensor2, attention_mask1=None, attention_mask2=None):
        cross_outputs1, cross_outputs2 = self.cross(input_tensor1, input_tensor2, attention_mask1, attention_mask2)
        attention_output1 = self.output1(cross_outputs1[0], input_tensor1)
        attention_output2 = self.output2(cross_outputs2[0], input_tensor2)
        # add attentions if we output them
        outputs1 = (attention_output1,) + cross_outputs1[1:]
        outputs2 = (attention_output2,) + cross_outputs2[1:]
        return outputs1, outputs2


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MLPLayer(nn.Module):
    def __init__(self, config):
        super(MLPLayer, self).__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    def forward(self, attention_outputs):
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # add attentions if we output them
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.mlp1 = MLPLayer(config)
        self.mlp2 = MLPLayer(config)

    def forward(self, hidden_states1, hidden_states2, attention_mask1=None, attention_mask2=None):
        attention_outputs1, attention_outputs2 = self.attention(hidden_states1, hidden_states2, attention_mask1, attention_mask2)
        outputs1 = self.mlp1(attention_outputs1)
        outputs2 = self.mlp2(attention_outputs2)
        return outputs1, outputs2



class CrossBertEncoder(nn.Module):
    def __init__(self, config):
        super(CrossBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(
            config.num_hidden_layers)])
        
    def get_outputs(self, all_hidden_states, all_attentions, hidden_states):
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs
    
    def forward(self, hidden_states1, hidden_states2, extended_attention_mask1=None, extended_attention_mask2=None):
        all_hidden_states1 = ()
        all_hidden_states2 = ()
        all_attentions1 = ()
        all_attentions2 = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states1 = all_hidden_states1 + (hidden_states1,)
                all_hidden_states2 = all_hidden_states2 + (hidden_states2,)
                
            layer_outputs1, layer_outputs2 = layer_module(hidden_states1, hidden_states2, extended_attention_mask1, extended_attention_mask2)
            hidden_states1 = layer_outputs1[0]
            hidden_states2 = layer_outputs2[0]

            if self.output_attentions:
                all_attentions1 = all_attentions1 + (layer_outputs1[1],)
                all_attentions2 = all_attentions2 + (layer_outputs2[1],)
                
        outputs1 = self.get_outputs(all_hidden_states1, all_attentions1, hidden_states1)
        outputs2 = self.get_outputs(all_hidden_states2, all_attentions2, hidden_states2)

        return outputs1, outputs2



