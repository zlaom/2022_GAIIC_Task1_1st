import copy
import json
import random 
from tqdm import tqdm 
import torch 
from torch.utils.data import Dataset
import numpy as np 


# attribute finetuning dataset
class AttrMatchDataset(Dataset):
    '''generate positive and negative samples for attribute matching finetuning'''
    def __init__(self, input_filename, relation_dict_file):
        with open(relation_dict_file, 'r') as f:
            relation_dict = json.load(f)
        self.relation_dict = relation_dict
        self.all_list = list(relation_dict.keys())

        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['key_attr']: # 必须有属性
                        self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
        key_attr = item['key_attr']
        
        split_label = torch.ones(20)
        attr_mask = torch.zeros(20)
        for query, attr in key_attr.items():
            attr_index = split.index(attr) # 先找到属性的位置
            attr_mask[attr_index] = 1
            if random.random() < 0.5: # 负例
                attr_list = random.sample(self.relation_dict[attr]['similar_attr'], 1)[0]
                if len(attr_list) == 1:
                    new_attr = attr_list[0]
                else:
                    new_attr = random.sample(attr_list, 1)[0]
                split[attr_index] = new_attr
                split_label[attr_index] = 0 # 标签不匹配
            else:
                if random.random() < 0.1: # 正例增强
                    if self.relation_dict[attr]['equal_attr']:
                        new_attr = random.sample(self.relation_dict[attr]['equal_attr'], 1)[0]
                        split[attr_index] = new_attr

        return image, split, split_label, attr_mask
    





def attrmatch_collate_fn(batch):
    tensors = []
    splits = []
    labels = []
    masks = []
    for feature, split, split_label, attr_mask in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(split_label)
        masks.append(attr_mask)
    tensors = torch.stack(tensors)
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    return tensors, splits, labels, masks









    # 属性替换也根据频率决定的比率进行替换
# class AttrMatchProbaDataset(Dataset):
#     '''generate positive and negative samples for attribute matching finetuning'''
#     def __init__(self, input_filename, attr_dict_file, vocab_dict_file):
#         with open(attr_dict_file, 'r') as f:
#             attr_dict = json.load(f)
#         with open(vocab_dict_file, 'r') as f:
#             vocab_dict = json.load(f)
#         self.negative_dict, self.proba_negative_dict = self.get_negative_dict(attr_dict, vocab_dict)
#         # 提取数据
#         self.items = []
#         for file in input_filename.split(','):
#             with open(file, 'r') as f:
#                 for line in tqdm(f):
#                     item = json.loads(line)
#                     if item['match']['图文']: # 训练集图文必须匹配
#                         if item['key_attr']: # 必须有属性
#                             self.items.append(item)
                
#     def __len__(self):
#         return len(self.items)
    
#     def get_negative_dict(self, attr_dict, vocab_dict):
#         proba_negative_dict = {}
#         negative_dict = {}
#         for query, attr_list in attr_dict.items():
#             for attr in attr_list:
#                 l = attr_list.copy()
#                 l.remove(attr)
#                 negative_dict[attr] = l
#                 # 统计频率并计算概率
#                 proba_list = []
#                 for l_attr in l:
#                     proba_list.append(vocab_dict[l_attr])

#                 proba_list = np.array(proba_list)
#                 proba_negative_dict[attr] = proba_list / np.sum(proba_list)
                
#         return negative_dict, proba_negative_dict
        
#     def __getitem__(self, idx):
#         item = self.items[idx]
#         image = torch.tensor(item['feature'])
#         split = item['vocab_split']
#         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
#         key_attr = item['key_attr']
        
#         split_label = torch.ones(20)
#         attr_mask = torch.zeros(20)
#         for query, attr in key_attr.items():
#             attr_index = split.index(attr) # 先找到属性的位置
#             attr_mask[attr_index] = 1
#             if random.random() > 0.5:
#                 split[attr_index] = np.random.choice(self.negative_dict[attr], p=self.proba_negative_dict[attr])
#                 split_label[attr_index] = 0 # 标签不匹配

#         return image, split, split_label, attr_mask