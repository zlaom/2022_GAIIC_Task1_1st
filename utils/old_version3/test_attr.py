import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

gpus = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

for file_index in range(4):

    ckpt_path = 'output/train/attr/result_merge/shuffle/0l6lexp6_0.9387.pth'


    vocab_file = 'data/new_data/vocab/vocab.txt'
    # attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'
    # with open(attr_dict_file, 'r') as f:
    #     attr_dict = json.load(f)

    test_file = f'output/fusion/no_pos_{file_index}.txt'
    out_dir = 'output/fusion/predict'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_file = os.path.join(out_dir, f"no_pos_attr_{file_index}.txt")


    # fuse model 
    split_layers = 0
    fuse_layers = 6
    n_img_expand = 6

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
            ori_split = data['vocab_split']
            key_attr = data['key_attr']
            
            image = image[None, ].cuda()
            pred = {}
            pred['图文'] = 1
            for query, attr in key_attr.items():
                if attr not in ori_split:
                    print(data['title'])
                    print(data['vocab_split'])
                    print(data['key_attr'])
                    break
                split = [[attr]]
                with torch.no_grad():
                    logits = model(image, split)
                    logits = logits.squeeze(1).cpu()

                    logits = torch.sigmoid(logits)
                    # logits[logits>0.5] = 1
                    # logits[logits<=0.5] = 0
                
                pred[query] = logits.item()
            
            data['pred'] = pred
            del data['feature']
            
            rets.append(json.dumps(data, ensure_ascii=False)+'\n')

    with open(out_file, 'w') as f:
        f.writelines(rets)


