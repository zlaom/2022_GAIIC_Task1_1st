import torch 
import json 
from tqdm import tqdm 
from torch.utils.data import Dataset 
import copy 
import random 
import numpy as np 

class AttrSequenceDataset(Dataset):
    def __init__(self, input_data, relation_dict_file, attr2id_dict_file, is_train):
        self.is_train = is_train
        with open(relation_dict_file, 'r') as f:
            relation_dict = json.load(f)
        with open(attr2id_dict_file, 'r') as f:
            attr2id_dict = json.load(f)
        
        self.attr2id_dict = attr2id_dict
        self.relation_dict = relation_dict
        self.all_list = list(relation_dict.keys())
        
        # 提取数据
        if self.is_train:
            self.items = []
            for item in input_data:
                for query, attr in item['key_attr'].items():
                    new_item = {}
                    new_item['feature'] = item['feature']
                    new_item['key_attr'] = attr
                    self.items.append(new_item)
        else:
            self.items = input_data
        print(len(self.items))
        
    def __len__(self):
        return len(self.items)
    
    # standard
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        if self.is_train:
            # 随机选一个属性
            attr = item['key_attr']
            if random.random() < 0.5: # 替换，随机挑选一个词替换
                label = 0
                attr_list = random.sample(self.relation_dict[attr]['similar_attr'], 1)[0]
                if len(attr_list) == 1:
                    attr = attr_list[0]
                else:
                    attr = random.sample(attr_list, 1)[0]
            else: 
                label = 1
                if self.relation_dict[attr]['equal_attr']:
                    if random.random() < 0.1: # 正例增强
                        label = 1
                        attr = random.sample(self.relation_dict[attr]['equal_attr'], 1)[0]
        else:
            (query, attr), = item['key_attr'].items()
            label = item['match'][query]

        attr_id = self.attr2id_dict[attr]

        return image, attr_id, label


