import torch 
import json 
from tqdm import tqdm 
from torch.utils.data import Dataset 
import copy 
import random 
import numpy as np 


class FuseReplaceDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.get_negative_dict(attr_dict)
        self.attr_dict = attr_dict
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
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        if self.is_train:
            split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
            label = 1
            if random.random() < 0.55: # 替换，随机挑选一个词替换
                label = 0
                rep_idx = random.sample([i for i in range(len(split))], 1)
                for i in rep_idx:
                    word = split[i]
                    if word in self.negative_dict: # 如果是关键属性则属性替换
                        split[i] = random.sample(self.negative_dict[word], 1)[0]
                    else:
                        new_word = np.random.choice(self.words_list, p=self.proba_list)
                        # if new_word in self.all_attr:
                        #     new_word = random.sample(self.all_attr, 1)[0]
                        if new_word in split: # 之前忽略的一个bug
                            label = 1
                        else:
                            split[i] = new_word
        else:
            label = item['match']['图文']


    # 去重，去掉同query的替换词
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     key_attr = item['key_attr']
    #     attr_list = []
    #     for query, attr in key_attr.items():
    #         attr_list = attr_list + self.attr_dict[query]
    #     if self.is_train:
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         label = 1
    #         if random.random() < 0.55: # 替换，随机挑选一个词替换
    #             label = 0
    #             rep_idx = random.sample([i for i in range(len(split))], 1)
    #             for i in rep_idx:
    #                 word = split[i]
    #                 if word in self.negative_dict: # 如果是关键属性则属性替换
    #                     split[i] = random.sample(self.negative_dict[word], 1)[0]
    #                 else:
    #                     new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                     # if new_word in self.all_attr:
    #                     #     new_word = random.sample(self.all_attr, 1)[0]
    #                     if new_word in split or new_word in attr_list: # 之前忽略的一个bug
    #                         label = 1
    #                     else:
    #                         split[i] = new_word
    #     else:
    #         label = item['match']['图文']
            

    # 可能替换多个词
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     if self.is_train:
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         label = 1
    #         if random.random() < 0.55: # 替换，随机挑选一个词替换
    #             label = 0
    #             rep_list = [1,2]
    #             rep_proba = [0.5, 0.5]
    #             rep_num = np.random.choice(rep_list, p=rep_proba)
    #             # rep_num = 2
    #             if len(split) >= rep_num:
    #                 rep_idx = random.sample([i for i in range(len(split))], rep_num)
    #             else:
    #                 rep_idx = random.sample([i for i in range(len(split))], 1)
    #             for i in rep_idx:
    #                 word = split[i]
    #                 if word in self.negative_dict: # 如果是关键属性则属性替换
    #                     split[i] = random.sample(self.negative_dict[word], 1)[0]
    #                 else:
    #                     new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                     # if new_word in self.all_attr:
    #                     #     new_word = random.sample(self.all_attr, 1)[0]
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
    #                     if random.random() < 0.5:
    #                         new_word = np.random.choice(self.negative_dict[word])
    #                     else:
    #                         new_word = np.random.choice(self.all_attr)
    #                 else:
    #                     new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                 if new_word not in split: # 之前忽略的一个bug
    #                     split[i] = new_word
    #                     label = 0
    #     else:
    #         label = item['match']['图文']

        return image, split, label


class SingleAttrDataset(Dataset):
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





# ITM(image-text matching) finetune
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


# ITM(image-text matching) finetune with augmentation
class ITMAugDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file, vocab_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        with open(vocab_dict_file, 'r') as f:
            vocab_dict = json.load(f)
        self.get_negative_dict(attr_dict)
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
        self.gender_list = ['男','女','男装','女装','男士','女士']
        self.gender_dict = {'男':'女', '男装':'女装', '男士':'女士', '女':'男', '女装':'男装', '女士':'男士'}
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        ori_split = item['vocab_split']
        split = copy.deepcopy(ori_split) # 要做拷贝，否则会改变self.items的值
        label = item['match']['图文']
        
        if label == 1 and random.random() > 0.5: # 正样本以一定概率转化为负样本
            # roll点，决定用哪种生成方式
            r = random.random()
            if r < 0.5:
                # 属性替换， 每个属性都有一定的概率被替换
                key_attr = item['key_attr']
                for query, attr in key_attr.items():
                    attr_index = split.index(attr) # 先找到属性的位置
                    if random.random() > 0.5:
                        split[attr_index] = np.random.choice(self.negative_dict[attr])
                        label = 0
            elif r < 1:
                # 属性随机替换
                key_attr = item['key_attr']
                for query, attr in key_attr.items():
                    attr_index = split.index(attr) # 先找到属性的位置
                    if random.random() > 0.5:
                        new_attr = np.random.choice(self.all_attr) 
                        if new_attr not in split:
                            split[attr_index] = new_attr
                        label = 0
            elif r < 0.8:
                # 属性添加，添加一个随机属性
                new_attr = np.random.choice(self.all_attr) 
                if new_attr not in split:
                    split = split + [new_attr]
                    label = 0
            elif r < 0:
                # 男女，男女士，男女装替换
                for gender in self.gender_list:
                    if gender in ori_split:
                        gender_index = ori_split.index(gender)
                        split[gender_index] = self.gender_dict[gender]
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










# 一些不同的替换策略

    # 只用关键属性随机替换
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     key_attr = item['key_attr']
    #     if self.is_train:
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         label = 1
    #         if random.random() < 0.6: # 替换，随机挑选一个属性替换
    #             if key_attr:
    #                 query = random.sample(list(key_attr.keys()), 1)[0]
    #                 attr = key_attr[query]
    #                 attr_index = split.index(attr)
    #                 new_attr = np.random.choice(self.all_attr)
    #                 # new_attr = np.random.choice(self.negative_dict[attr])

    #                 if new_attr not in split: # 之前忽略的一个bug
    #                     split[attr_index]  = new_attr
    #                     label = 0
    #     else:
    #         label = item['match']['图文']


    # 只做词替换，不做属性替换
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     if self.is_train:
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         label = 1
    #         if random.random() < 0.5: # 替换，随机挑选一个词替换
    #             label = 0
    #             rep_idx = random.sample([i for i in range(len(split))], 1)
    #             for i in rep_idx:
    #                 new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                 if new_word in split: # 之前忽略的一个bug
    #                     label = 1
    #                 else:
    #                     split[i] = new_word
    #     else:
    #         label = item['match']['图文']


    # 有概率添加随机词
    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = torch.tensor(item['feature'])
    #     split = item['vocab_split']
    #     if self.is_train:
    #         label = 1
    #         split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
    #         if random.random() < 0.4: # 替换，随机挑选一个词替换
    #             rep_idx = random.sample([i for i in range(len(split))], 1)
    #             for i in rep_idx:
    #                 word = split[i]
    #                 if word in self.negative_dict: # 如果是关键属性则属性替换
    #                     split[i] = random.sample(self.negative_dict[word], 1)[0]
    #                     label = 0
    #                 else:
    #                     new_word = np.random.choice(self.words_list, p=self.proba_list)
    #                     # new_word = np.random.choice(self.words_list)
    #                     if new_word not in split: # 之前忽略的一个bug
    #                         split[i] = new_word
    #                         label = 0
    #         if random.random() < 0.2: # 加词，随机加一个词到末尾
    #             if random.random() < 0.25: # 增加了来自于属性的概率
    #                 new_word = random.sample(list(self.negative_dict.keys()), 1)[0]
    #             else:
    #                 new_word = np.random.choice(self.words_list, p=self.proba_list)
    #             if new_word not in split: # 之前忽略的一个bug
    #                 split = split + [new_word]
    #                 label = 0
    #     else:
    #         label = item['match']['图文']