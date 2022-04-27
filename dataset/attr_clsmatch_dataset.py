from xml.dom.minidom import Attr
import torch 
import json 
from tqdm import tqdm 
from torch.utils.data import Dataset 
import copy 
import random 
import numpy as np 

class AttrClsMatchDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.get_negative_dict(attr_dict)
        
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
    
    def get_negative_dict(self, attr_dict):
        all_attr = []
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            all_attr = all_attr + attr_list
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        self.negative_dict = negative_dict
        self.all_attr = all_attr
        
        
        

class FuseReplaceDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.get_negative_dict(attr_dict)
        
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
                    self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def get_negative_dict(self, attr_dict):
        all_attr = []
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            all_attr = all_attr + attr_list
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        self.negative_dict = negative_dict
        self.all_attr = all_attr
    
    # 标准fusereplace
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     if self.is_train:
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         label = 1
    #         if random.random() < 0.6: # 替换，随机挑选一个词替换
    #             label = 0
    #             rep_idx = random.sample([i for i in range(len(split))], 1)
    #             for i in rep_idx:
    #                 word = split[i]
    #                 if word in self.negative_dict: # 如果是关键属性则属性替换
    #                     split[i] = random.sample(self.negative_dict[word], 1)[0]
    #                 else:
    #                     # new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                     new_word = np.random.choice(self.words_list)
    #                     if new_word in split: # 之前忽略的一个bug
    #                         label = 1
    #                     else:
    #                         split[i] = new_word
    #     else:
    #         label = item['match']['图文']


    # 属性随机替换的fusereplace
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     if self.is_train:
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         label = 1
    #         if random.random() < 0.55: # 替换，随机挑选一个词替换
    #             rep_idx = random.sample([i for i in range(len(split))], 1)
    #             for i in rep_idx:
    #                 word = split[i]
    #                 if word in self.all_attr: # 如果是关键属性则属性替换
    #                     new_word = np.random.choice(self.all_attr)
    #                 else:
    #                     new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                 if new_word not in split: # 之前忽略的一个bug
    #                     split[i] = new_word
    #                     label = 0
    #     else:
    #         label = item['match']['图文']


    # 属性随机替换的fusereplace，提高替换同类型的概率
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            if random.random() < 0.55: # 替换，随机挑选一个词替换
                rep_idx = random.sample([i for i in range(len(split))], 1)
                for i in rep_idx:
                    word = split[i]
                    if word in self.all_attr: # 如果是关键属性则属性替换
                        if random.random() < 0.5:
                            new_word = np.random.choice(self.negative_dict[word])
                        else:
                            new_word = np.random.choice(self.all_attr)
                    else:
                        new_word = np.random.choice(self.words_list, p=self.proba_list)
                    if new_word not in split: # 之前忽略的一个bug
                        split[i] = new_word
                        label = 0
        else:
            label = item['match']['图文']

        return image, split, label



# 子类没有重写__init__,会自动调用父类__init__
class SingleAttrDataset(AttrClsMatchDataset):
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        key_attr = item['key_attr']
        
        query = np.random.choice(list(key_attr.keys()))
        attr = key_attr[query]

        if random.random() < 0.5: # 替换，随机挑选一个词替换
            split = random.sample(self.negative_dict[attr], 1)
            label = 0
        else:
            split = [attr]
            label = 1

        return image, split, label


# random select a word from split as true attr, or replace it by a random word as false attr
class RandomReplaceDataset(AttrClsMatchDataset):
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        ori_split = item['vocab_split']
        
        split = [np.random.choice(ori_split)]
        label = 1
        if random.random() < 0.55: 
            new_word = np.random.choice(self.words_list, p=self.proba_list)
            if new_word not in ori_split:
                split = [new_word]
                label = 0

        return image, split, label


# fuse method of random replace and key attr replace
class FuseReplaceDataset(AttrClsMatchDataset):
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        ori_split = item['vocab_split']
        key_attr = item['key_attr']
        
        query = np.random.choice(list(key_attr.keys()))
        attr = key_attr[query]

        if random.random() < 0.3: # 属性替换
            if random.random() < 0.5: 
                split = random.sample(self.negative_dict[attr], 1)
                label = 0
            else:
                split = [attr]
                label = 1
        else: # 随机挑词替换
            split = [np.random.choice(ori_split)]
            label = 1
            if random.random() < 0.55: 
                new_word = np.random.choice(self.words_list, p=self.proba_list)
                if new_word not in ori_split:
                    split = [new_word]
                    label = 0

                

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

