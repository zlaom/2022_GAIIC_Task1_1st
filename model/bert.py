import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('hfl/chinese-macbert-base', cache_dir='data/pretrained_model/macbert_base')
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base', cache_dir='data/pretrained_model/macbert_base')
        self.classifier = nn.Sequential(
            nn.Linear(768+2048, 768),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 1)
        )
    def forward(self, image_features, texts):
        tokens = self.tokenizer(texts,
                                padding=True, 
                                truncation=True, 
                                # max_length=40, 
                                return_tensors='pt')
        input_ids = tokens['input_ids'].cuda()
        token_type_ids = tokens['token_type_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()

        bert_hidden = self.bert(input_ids=input_ids, 
                      token_type_ids=token_type_ids, 
                      attention_mask=attention_mask)[0]

        title_features = bert_hidden[:,0,:]
        features = torch.cat([image_features, title_features], dim=1)
        x = self.classifier(features)
        return x