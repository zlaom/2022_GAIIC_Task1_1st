import torch
import torch.nn as nn
from model.split_bert.splitbert import SplitBertModel
from model.bert.tokenizer import Tokenizer

class PretrainSplitBert(nn.Module):
    def __init__(self, config, vocab_file):
        super().__init__()
        self.bert = SplitBertModel(config)
        self.tokenizer = Tokenizer(vocab_file)
        
        img_dim = 2048
        bert_dim = config.hidden_size
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.LayerNorm(img_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + img_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bert_dim, 1)
        )
        

    def forward(self, image_features, splits): # 注意splits需要为二维列表
        image_features = self.image_encoder(image_features)
        tokens = self.tokenizer(splits)
        
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        seq_len = sequence_output.shape[1]
        image_features = image_features[:, None, :].repeat(1, seq_len, 1)

        features = torch.cat([image_features, sequence_output], dim=-1)

        x = self.classifier(features)
        return x, attention_mask
