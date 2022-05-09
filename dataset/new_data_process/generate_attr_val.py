import os
import random
import torch 
import json
import numpy as np 
from tqdm import tqdm 
import copy 

seed = 0
random.seed(seed)
np.random.seed(seed)


val_file = 'data/new_data/divided/attr/fine5000.txt'
vocab_dict_file = 'data/new_data/vocab/vocab_dict.json'
vocab_file = 'data/new_data/vocab/vocab.txt'
relation_dict_file = 'data/new_data/equal_processed_data/attr_relation_dict.json'
save_file = 'data/new_data/divided/attr/val/fine5000_0.25posaug.txt'
with open(relation_dict_file, 'r') as f:
    relation_dict = json.load(f)
    

rets = []
with open(val_file, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        if item['key_attr']: # 必须有属性
            for query, attr in item['key_attr'].items():
                key_attr = {}
                match = {}
                new_item = copy.deepcopy(item)
                if random.random() < 0.5: # 替换，随机挑选一个词替换
                    label = 0
                    attr_list = random.sample(relation_dict[attr]['similar_attr'], 1)[0]
                    if len(attr_list) == 1:
                        split = attr_list
                    else:
                        attr = random.sample(attr_list, 1)[0]
                        split = [attr]
                else: 
                    label = 1
                    split = [attr]
                    if relation_dict[attr]['equal_attr']:
                        if random.random() < 0.25: # 正例增强
                            label = 1
                            split = random.sample(relation_dict[attr]['equal_attr'], 1)
                            
                key_attr[query] = split[0]
                match[query] = label
                new_item['key_attr'] = key_attr
                new_item['match'] = match
                rets.append(json.dumps(new_item, ensure_ascii=False)+'\n')
        
with open(save_file, 'w') as f:
    f.writelines(rets)