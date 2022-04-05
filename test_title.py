import os
import itertools
import torch 
import json
import numpy as np 
import random 
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel


gpus = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 2
fuse_layers = 4
n_img_expand = 2
vocab_file = 'dataset/vocab/vocab.txt'

attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'
test_file = 'data/equal_split_word/test4000.txt'
ckpt_path = 'output/title_finetune/keyattrmatch/0.9328.pth'
out_file = "pred_title.txt"

with open(attr_dict_file, 'r') as f:
    attr_dict = json.load(f)
    
# model
split_config = BertConfig(num_hidden_layers=split_layers)
fuse_config = BertConfig(num_hidden_layers=fuse_layers)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
model.load_state_dict(torch.load(ckpt_path))
model.cuda()


# test
model.eval()
rets = []
with open(test_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        image = data['feature']
        image = torch.tensor(image)
        split = data['vocab_split']
        key_attr = data['key_attr']
        
        image = image[None, ].cuda()
        split = [split]

        with torch.no_grad():
            logits = model(image, split)
            logits = logits.squeeze(1).cpu()

            logits = torch.sigmoid(logits)
            logits[logits>0.5] = 1
            logits[logits<=0.5] = 0
            
        match = {}
        match['图文'] = int(logits.item())
        for query, attr in key_attr.items():
            match[query] = 1
        
        ret = {"img_name": data["img_name"],
            "match": match
        }
        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')

with open(out_file, 'w') as f:
    f.writelines(rets)


