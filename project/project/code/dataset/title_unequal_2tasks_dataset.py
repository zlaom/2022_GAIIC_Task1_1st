import torch 
import json 
from tqdm import tqdm 
from torch.utils.data import Dataset 
import copy 
import random 
import numpy as np 


class FuseReplaceDataset(Dataset):
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
            # if word not in self.all_attr:
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
    
    # 标准fusereplace
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        key_attr = item['key_attr']
        
        split_label = torch.ones(20)
        equal_list = []
        for query, attr in key_attr.items():
            equal_list.extend(self.relation_dict[attr]['equal_attr'])
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            rep_num = random.sample([1,2], 1)[0]
            if len(split) < rep_num:
                rep_num = 1
            # rep_num = 1
            if random.random() < 0.6: # 替换，随机挑选一个词替换
                label = 0
                rep_idx = random.sample([i for i in range(len(split))], rep_num)
                for i in rep_idx:
                    word = split[i]
                    if word in self.relation_dict: # 如果是关键属性则属性替换
                        attr_list = random.sample(self.relation_dict[word]['similar_attr'], 1)[0]
                        if len(attr_list) == 1:
                            new_attr = attr_list[0]
                        else:
                            new_attr = random.sample(attr_list, 1)[0]
                        split[i] = new_attr
                        split_label[i] = 0
                    else: # 否则，随机替换
                        new_word = np.random.choice(self.words_list, p=self.proba_list)
                        if new_word in split or new_word in equal_list:
                            label = 1
                        else:
                            label = 0
                            split[i] = new_word
                            split_label[i] = 0
        else:
            label = item['match']['图文']

        return image, split, label, split_label
    


class DiscreteFuseReplaceDataset(Dataset):
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
            # if word not in self.all_attr:
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
    
    # 标准fusereplace
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        key_attr = item['key_attr']
        
        split_label = torch.ones(20)
        equal_list = []
        for query, attr in key_attr.items():
            equal_list.extend(self.relation_dict[attr]['equal_attr'])
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            rep_num = random.sample([1,2], 1)[0]
            if len(split) < rep_num:
                rep_num = 1
            # rep_num = 1
            rep_idx = random.sample([i for i in range(len(split))], rep_num)
            for i in rep_idx:
                word = split[i]
                if word in self.relation_dict: # 如果是关键属性则属性替换
                    if random.random() < 0.5: # 0.4, 0.6, 0.8
                        label = 0
                        attr_list = random.sample(self.relation_dict[word]['similar_attr'], 1)[0]
                        if len(attr_list) == 1:
                            new_attr = attr_list[0]
                        else:
                            new_attr = random.sample(attr_list, 1)[0]
                        split[i] = new_attr
                        split_label[i] = 0
                else: # 否则，随机替换
                    if random.random() < 0.5: # 0.7, 0.6, 0.5
                        new_word = np.random.choice(self.words_list, p=self.proba_list)
                        if new_word not in split and new_word not in equal_list:
                            label = 0
                            split[i] = new_word
                            split_label[i] = 0
        else:
            label = item['match']['图文']

        return image, split, label, split_label



def cls_collate_fn(batch):
    tensors = []
    splits = []
    labels = []
    split_labels = []
    
    for feature, split, label, split_label in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(label)
        split_labels.append(split_label)
        
    tensors = torch.stack(tensors)
    labels = torch.tensor(labels)
    split_labels = torch.stack(split_labels)
    return tensors, splits, labels, split_labels