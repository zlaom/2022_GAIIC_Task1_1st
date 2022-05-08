import torch 
import json 
from tqdm import tqdm 
from torch.utils.data import Dataset 
import copy 
import random 
import numpy as np 
import itertools

class FuseReplaceDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.negative_dict = self.get_negative_dict(attr_dict)
        # {x:neg[]}

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
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        return negative_dict

    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            if random.random() < 0.6:
                label = 0
                if random.random() > 0.5:
                    rep_idx = random.sample([i for i in range(len(split))], 1)
                else:
                    if len(split) > 2:
                        rep_idx = random.sample([i for i in range(len(split))], 2)
                    else:
                        rep_idx = random.sample([i for i in range(len(split))], 1)
                for i in rep_idx:
                    word = split[i]
                    if word in self.negative_dict: # 如果是关键属性则属性替换
                        split[i] = random.sample(self.negative_dict[word], 1)[0]
                    else:
                        new_word = np.random.choice(self.words_list, p=self.proba_list)
                        if new_word in split: # 之前忽略的一个bug
                            label = 1
                        else:
                            split[i] = new_word
            else:
                label = 1
        else:
            label = item['match']['图文']

        return image, split, label




class NewFuseReplaceDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.positive_dict, self.negative_dict = self.get_positive_negative_dict(attr_dict)
        # {x:neg[]}

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
    
    def get_positive_negative_dict(self, attr_dict):
        negative_dict = {}
        positive_dict = {}
        for query, attr_list in attr_dict.items():
            for i, attrs in enumerate(attr_list):
                for attr in attrs:
                    l = attr_list.copy()
                    l.pop(i)
                    negative_dict[attr] = list(itertools.chain.from_iterable(l))
                    positive_dict[attr] = attrs
        return positive_dict, negative_dict

    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            if random.random() < 0.6:
                label = 0
                if random.random() > 0.5:
                    rep_idx = random.sample([i for i in range(len(split))], 1)
                else:
                    if len(split) > 2:
                        rep_idx = random.sample([i for i in range(len(split))], 2)
                    else:
                        rep_idx = random.sample([i for i in range(len(split))], 1)
                for i in rep_idx:
                    word = split[i]
                    if word in self.negative_dict: # 如果是关键属性则属性替换
                        split[i] = random.sample(self.negative_dict[word], 1)[0]
                    else:
                        new_word = np.random.choice(self.words_list, p=self.proba_list)
                        flag = False
                        for attr in item['key_attr'].values():
                            if new_word in self.positive_dict[attr]:
                                flag = True
                                break
                        if new_word in split: # 之前忽略的一个bug
                            flag = True
                            
                        if flag:
                            label = 1
                        else:
                            split[i] = new_word
            else:
                label = 1
        else:
            label = item['match']['图文']

        return image, split, label


class FuseAttrReplaceDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.attr_dict = attr_dict
        self.negative_dict = self.get_negative_dict(attr_dict)
        # {x:neg[]}

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
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        return negative_dict
    
    def get_dis_attr(self, keys):
        all_keys = list(self.attr_dict.keys())
        key = np.random.choice(all_keys)
        while key in keys:
            key = np.random.choice(all_keys)
        new_word = np.random.choice(self.attr_dict[key])
        return new_word

    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            if random.random() < 0.6:
                label = 0
                
                rep_idx = random.sample([i for i in range(len(split))], 1)
                for i in rep_idx:
                    word = split[i]
                    if word in self.negative_dict: # 如果是关键属性则属性替换
                        split[i] = random.sample(self.negative_dict[word], 1)[0]
                    else:
                        if item['key_attr'] is not None:
                            keys = item['key_attr'].keys()
                        else:
                            keys = {}
                        new_word = self.get_dis_attr(keys)
                        split[i] = new_word

            else:
                label = 1
        else:
            label = item['match']['图文']

        return image, split, label


# ITM(image-text matching) finetune
class ITMAttrDataset(Dataset):
    def __init__(self, input_filename):
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    self.items.append(item)
                
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = list(item['key_attr'].values())
        label = item['match']['图文']

        return image, split, label

class ITMDataset(Dataset):
    def __init__(self, input_filename):
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    self.items.append(item)
                
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        label = item['match']['图文']

        return image, split, label

class SplitITMDataset(Dataset):
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        label = item['match']['图文']

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