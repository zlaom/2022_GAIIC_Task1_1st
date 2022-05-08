import torch
import torch.nn as nn
from models.split_bert.split_bert import SplitBertModel
from models.hero_bert.tokenizer import Tokenizer

class SESplitBert(nn.Module):
    def __init__(self, config, vocab_file):
        super().__init__()
        self.bert = SplitBertModel(config)
        self.tokenizer = Tokenizer(vocab_file)
        
        img_dim = 2048
        bert_dim = config.hidden_size
        self.image_encoder = nn.Sequential(
            nn.Linear(img_dim, bert_dim),
            nn.LayerNorm(bert_dim)
        )

        self.image_wsa = nn.Sequential(
            nn.Linear(bert_dim * 2, bert_dim),
            nn.BatchNorm1d(bert_dim),
            nn.Sigmoid() # change 
        )
        self.text_wsa = nn.Sequential(
            nn.Linear(bert_dim * 2, bert_dim),
            nn.BatchNorm1d(bert_dim),
            nn.Sigmoid()
        )
        
        self.itm_head = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 2),
        )
        

    def forward(self, image, splits): # 注意splits需要为二维列表
        image = self.image_encoder(image)
        tokens = self.tokenizer(splits)
        
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        text = text[:, 0, :]
        features = torch.cat((image, text), dim=-1)
        image = torch.nn.functional.normalize(image, p=2, dim=-1)
        text = torch.nn.functional.normalize(text, p=2, dim=-1)
        image_w = self.image_wsa(features)
        text_w = self.text_wsa(features)
        fusion_feature = image * image_w + text * text_w

        logits = self.itm_head(fusion_feature)
        return logits
