import copy
import json
import random 
from tqdm import tqdm 
import torch 
from torch.utils.data import Dataset
import numpy as np 


class WordReplaceDataset(Dataset):
    def __init__(self, input_filename, vocab_dict):
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
                    if item['match']['图文']: # 训练集图文必须匹配
                        self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        ori_split = self.items[idx]['vocab_split']
        split = copy.deepcopy(ori_split) # 要做拷贝，否则会改变self.items的值
        
        split_label = torch.ones(20)
        for i, word in enumerate(split):
            if random.random() > 0.5: # 替换
                new_word = np.random.choice(self.words_list, p=self.proba_list)
                if new_word not in ori_split and new_word not in split: # 修复之前忽略的bug
                    split[i] = new_word
                    split_label[i] = 0

        return image, split, split_label
    

# 属性会进行属性替换而非随机替换
class FuseReplaceDataset(Dataset):
    def __init__(self, input_filename, vocab_dict, attr_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.negative_dict = self.get_negative_dict(attr_dict)
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
                    if item['match']['图文']: # 训练集图文必须匹配
                        self.items.append(item)
    
    def get_negative_dict(self, attr_dict):
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        return negative_dict
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        ori_split = self.items[idx]['vocab_split']
        split = copy.deepcopy(ori_split) # 要做拷贝，否则会改变self.items的值
        
        split_label = torch.ones(20)
        for i, word in enumerate(split):
            if random.random() > 0.5: # 替换
                if word in self.negative_dict:
                    split[i] = random.sample(self.negative_dict[word], 1)[0]
                    split_label[i] = 0
                else:
                    new_word = np.random.choice(self.words_list, p=self.proba_list)
                    if new_word not in ori_split and new_word not in split: # 修复之前忽略的bug
                        split[i] = new_word
                        split_label[i] = 0

        return image, split, split_label
    

# 属性替换也根据频率决定的比率进行替换
class FuseProbaReplaceDataset(Dataset):
    def __init__(self, input_filename, vocab_dict, attr_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.negative_dict, self.proba_negative_dict = self.get_negative_dict(attr_dict, vocab_dict)
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
                    if item['match']['图文']: # 训练集图文必须匹配
                        self.items.append(item)
    
    def get_negative_dict(self, attr_dict, vocab_dict):
        proba_negative_dict = {}
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
                # 统计频率并计算概率
                proba_list = []
                for l_attr in l:
                    proba_list.append(vocab_dict[l_attr])

                proba_list = np.array(proba_list)
                proba_negative_dict[attr] = proba_list / np.sum(proba_list)
                
        return negative_dict, proba_negative_dict
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        ori_split = self.items[idx]['vocab_split']
        split = copy.deepcopy(ori_split) # 要做拷贝，否则会改变self.items的值
        
        split_label = torch.ones(20)
        for i, word in enumerate(split):
            if random.random() > 0.5: # 替换
                if word in self.negative_dict:
                    split[i] = np.random.choice(self.negative_dict[word], p=self.proba_negative_dict[word])
                    split_label[i] = 0
                else:
                    new_word = np.random.choice(self.words_list, p=self.proba_list)
                    if new_word not in ori_split and new_word not in split: # 修复之前忽略的bug
                        split[i] = new_word
                        split_label[i] = 0

        return image, split, split_label
    
    
def word_collate_fn(batch):
    tensors = []
    splits = []
    labels = []
    for feature, split, split_label in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(split_label)
    tensors = torch.stack(tensors)
    labels = torch.stack(labels)
    return tensors, splits, labels