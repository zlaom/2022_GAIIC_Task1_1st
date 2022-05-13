import torch 
import json 
from tqdm import tqdm 
from torch.utils.data import Dataset 
import copy 
import random 
import numpy as np 

class AttrClsMatchDataset(Dataset):
    def __init__(self, input_filename, relation_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(relation_dict_file, 'r') as f:
            relation_dict = json.load(f)
            
        self.relation_dict = relation_dict
        self.all_list = list(relation_dict.keys())
        # 取出所有可替换的词及出现的次数比例
        words_list = []
        proba_list = []
        for word, n in vocab_dict.items():
            words_list.append(word)
            proba_list.append(n)
        self.words_list = words_list 
        proba_list = np.array(proba_list)
        self.proba_list = proba_list / np.sum(proba_list)
        
        
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
        
        

class SingleAttrDataset(AttrClsMatchDataset):
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        key_attr = item['key_attr']
        
        if self.is_train:
            # 随机选一个属性
            query = np.random.choice(list(key_attr.keys()))
            attr = key_attr[query]

            if random.random() < 0.5: # 替换，随机挑选一个词替换
                label = 0
                attr_list = random.sample(self.relation_dict[attr]['similar_attr'], 1)[0]
                if len(attr_list) == 1:
                    split = attr_list
                else:
                    attr = random.sample(attr_list, 1)[0]
                    split = [attr]
            else: 
                label = 1
                split = [attr]
                # if self.is_train:
                #     if self.relation_dict[attr]['equal_attr']:
                #         if random.random() < 0.1: # 正例增强
                #             label = 1
                #             split = random.sample(self.relation_dict[attr]['equal_attr'], 1)
        else:
            (query, attr), = key_attr.items()
            split = [attr]
            label = item['match'][query]
            
        return image, split, label



class AttrSequenceDataset(Dataset):
    def __init__(self, input_filename, relation_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(relation_dict_file, 'r') as f:
            relation_dict = json.load(f)
            
        self.relation_dict = relation_dict
        self.all_list = list(relation_dict.keys())
        # 取出所有可替换的词及出现的次数比例
        words_list = []
        proba_list = []
        for word, n in vocab_dict.items():
            words_list.append(word)
            proba_list.append(n)
        self.words_list = words_list 
        proba_list = np.array(proba_list)
        self.proba_list = proba_list / np.sum(proba_list)
        
        
        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if self.is_train:
                        if item['key_attr']: # 必须有属性
                            for query, attr in item['key_attr'].items():
                                new_item = copy.deepcopy(item)
                                new_item['key_attr'] = attr
                                self.items.append(new_item)
                    else:
                        self.items.append(item)
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
                    if random.random() < 0.25: # 正例增强
                        label = 0.8
                        attr = random.sample(self.relation_dict[attr]['equal_attr'], 1)[0]
        else:
            (query, attr), = item['key_attr'].items()
            label = item['match'][query]
        split = [attr]
         
        return image, split, label



    # 句子输入
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     ori_split = item['vocab_split']
    #     if self.is_train:
    #         attr = item['key_attr']
    #         if random.random() < 0.5: # 替换
    #             label = 0
    #             attr_list = random.sample(self.relation_dict[attr]['similar_attr'], 1)[0]
    #             if len(attr_list) == 1:
    #                 split = attr_list
    #             else:
    #                 attr = random.sample(attr_list, 1)[0]
    #                 split = [attr]
    #         else: 
    #             label = 1
    #             split = [attr]
    #             if self.relation_dict[attr]['equal_attr']:
    #                 if random.random() < 0.25: # 正例增强
    #                     label = 0.8
    #                     split = random.sample(self.relation_dict[attr]['equal_attr'], 1)
    #     else:
    #         (query, attr), = item['key_attr'].items()
    #         split = [attr]
    #         label = item['match'][query]
    #     # 添加所有非属性词
    #     for word in ori_split:
    #         if word not in self.all_list:
    #             split.append(word)
    #     return image, split, label
















class FuseReplaceDataset(AttrClsMatchDataset):
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        ori_split = item['vocab_split']
        key_attr = item['key_attr']
        
        query = np.random.choice(list(key_attr.keys()))
        attr = key_attr[query]

        if random.random() < 0.5: # 替换
            if random.random() < 0.8:
                # 同query替换
                new_attr = random.sample(self.negative_dict[attr], 1)[0]
                split = [new_attr]
                label = 0
            else:
                # 随机属性替换
                new_attr = np.random.choice(self.all_attr, p=self.all_attr_proba)
                # new_attr = random.sample(self.all_attr, 1)[0]
                if new_attr not in ori_split:
                    split = [new_attr]
                    label = 0
                else:
                    split = [attr]
                    label = 1
        else:
            split = [attr]
            label = 1

        return image, split, label



def cls_collate_fn(batch):
    tensors = []
    splits = []
    labels = []

    for feature, split, label in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(label)

    tensors = torch.stack(tensors)
    labels = torch.tensor(labels)

    return tensors, splits, labels





















    # 异类增强
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
        
    #     if self.is_train:
    #         # 随机选一个属性
    #         attr = item['key_attr']
    #         if random.random() < 0.5: # 替换，随机挑选一个词替换
    #             label = 0
    #             if random.random() < 0.1:
    #                 attr_list = random.sample(self.relation_dict[attr]['unsimilar_attr'], 1)[0]
    #             else:
    #                 attr_list = random.sample(self.relation_dict[attr]['similar_attr'], 1)[0]
    #             if len(attr_list) == 1:
    #                 split = attr_list
    #             else:
    #                 attr = random.sample(attr_list, 1)[0]
    #                 split = [attr]
    #         else: 
    #             label = 1
    #             split = [attr]
    #             if self.relation_dict[attr]['equal_attr']:
    #                 if random.random() < 0.25: # 正例增强
    #                     label = 0.8
    #                     split = random.sample(self.relation_dict[attr]['equal_attr'], 1)
    #     else:
    #         (query, attr), = item['key_attr'].items()
    #         split = [attr]
    #         label = item['match'][query]
            
    #     return image, split, label