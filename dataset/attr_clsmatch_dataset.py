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
        
        # 取出所有可替换的词及出现的次数比例
        words_list = []
        proba_list = []
        for word, n in vocab_dict.items():
            words_list.append(word)
            proba_list.append(n)
        self.words_list = words_list 
        proba_list = np.array(proba_list)
        self.proba_list = proba_list / np.sum(proba_list)
        
        self.get_negative_dict(attr_dict, vocab_dict)
        
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
    
    def get_negative_dict(self, attr_dict, vocab_dict):
        all_attr = []
        negative_dict = {}
        negative_dict_proba = {}
        for query, attr_list in attr_dict.items():
            all_attr = all_attr + attr_list
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
                # proba
                # proba_list = []
                # for _attr in l:
                #     proba_list.append(vocab_dict[_attr])
                # proba_list = np.array(proba_list)
                # proba_list = proba_list / np.sum(proba_list)
                # negative_dict_proba[attr] = proba_list
                
        # negative dict
        # self.negative_dict_proba = negative_dict_proba
        self.negative_dict = negative_dict
        # all attr
        self.all_attr = all_attr
        all_attr_proba = []
        for attr in all_attr:
            all_attr_proba.append(vocab_dict[attr])
        all_attr_proba = np.array(all_attr_proba)
        self.all_attr_proba = all_attr_proba / np.sum(all_attr_proba)
        
        

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

