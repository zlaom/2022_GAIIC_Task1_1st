import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.splitmodel import PretrainSplitBert
from model.crossmodel import PretrainCrossModel
from model.fusemodel import FuseModel
from model.fusecrossmodel import FuseCrossModel, FuseCrossModelWithFusehead

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

gpus = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

vocab_file = 'dataset/vocab/vocab.txt'
test_file = 'data/equal_split_word/test10000.txt'
attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'
out_file = "pred_attr_B.txt"

with open(attr_dict_file, 'r') as f:
    attr_dict = json.load(f)

ckpt_path = 'output/split_finetune/attr/final_bert/0l12lexp6/0.9356.pth'

# fuse model
# num_hidden_layers = 6
# ckpt_path = 'output/attr_finetune/base_pretrain0.8771/0.9424.pth'

# config = BertConfig(num_hidden_layers=num_hidden_layers)
# model = PretrainSplitBert(config, vocab_file)
# model.load_state_dict(torch.load(ckpt_path))
# model.cuda()


# cross model
# split_layers = 5
# cross_layers = 1
# n_img_expand = 2

# split_config = BertConfig(num_hidden_layers=split_layers)
# cross_config = BertConfig(num_hidden_layers=cross_layers)
# model = PretrainCrossModel(split_config, cross_config, vocab_file, n_img_expand=n_img_expand)
# model.load_state_dict(torch.load(ckpt_path))
# model.cuda()

# fuse model 
split_layers = 0
fuse_layers = 12
n_img_expand = 6

split_config = BertConfig(num_hidden_layers=split_layers)
fuse_config = BertConfig(num_hidden_layers=fuse_layers)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
model.load_state_dict(torch.load(ckpt_path))
model.cuda()

# fuse cross model
# split_config = BertConfig(num_hidden_layers=split_layers)
# fuse_config = BertConfig(num_hidden_layers=fuse_layers)
# cross_config = BertConfig(num_hidden_layers=cross_layers)
# model = FuseCrossModel(split_config, fuse_config, cross_config, vocab_file, n_img_expand=n_img_expand)
# model.load_state_dict(torch.load(ckpt_path))
# model.cuda()

# fuse cross model with fuse head
# split_config = BertConfig(num_hidden_layers=split_layers)
# fuse_config = BertConfig(num_hidden_layers=fuse_layers)
# cross_config = BertConfig(num_hidden_layers=cross_layers)
# model = FuseCrossModelWithFusehead(split_config, fuse_config, cross_config, vocab_file, n_img_expand=n_img_expand)
# model.load_state_dict(torch.load(ckpt_path))
# model.cuda()


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
        attr_mask = torch.zeros(20)
        query_seq = {} # 整理query对应最后logits的顺序
        for query, attr in key_attr.items():
            if attr not in split:
                print(data['title'])
                print(data['vocab_split'])
                print(data['key_attr'])
                break
            attr_index = split.index(attr) # 先找到属性的位置
            attr_mask[attr_index] = 1
            query_seq[query] = attr_index
            
        attr_mask = attr_mask[None, ]
        split = [split]
        image = image[None, ].cuda()
        

        with torch.no_grad():
            logits, mask = model(image, split, word_match=True)
            logits = logits.squeeze(2).cpu()
            
            _, W = logits.shape
            attr_mask = attr_mask[:, :W]
            
            mask = mask.to(torch.bool)
            attr_mask = attr_mask.to(torch.bool)
            attr_mask = attr_mask[mask]
            logits = logits[mask]

            logits = torch.sigmoid(logits)
            logits[logits>0.5] = 1
            logits[logits<=0.5] = 0
            
        match = {}
        match['图文'] = 1
        for query, index in query_seq.items():
            match[query] = int(logits[index])
        
        ret = {"img_name": data["img_name"],
            "match": match
        }
        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')

with open(out_file, 'w') as f:
    f.writelines(rets)


