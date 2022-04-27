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
    def __init__(self, input_filename, attr_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.negative_dict = self.get_negative_dict(attr_dict)
        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        if item['key_attr']: # 必须有属性
                            self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def get_negative_dict(self, attr_dict):
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            negative_dict[query] = {}
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[query][attr] = l
        return negative_dict
        
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
            if random.random() > 0.5:
                new_attr = random.sample(self.negative_dict[query][attr], 1)[0]
                split[attr_index] = new_attr
                split_label[attr_index] = 0 # 标签不匹配

        return image, split, split_label, attr_mask


# attribute finetuning title concate attr match dataset
class TitleCatAttrMatchDataset(Dataset):
    '''generate positive and negative samples for attribute matching finetuning'''
    def __init__(self, input_filename, neg_attr_dict_file):
        with open(neg_attr_dict_file, 'r') as f:
            self.neg_attr_dict = json.load(f)
        # 提取数据
        self.items = []
        i = 0
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        if item['key_attr']: # 必须有属性
                            # 生成所有离散属性
                            for key, value in item['key_attr'].items():
                                new_item = copy.deepcopy(item)
                                # 删除其他属性
                                new_item['key_attr'] = value
                                remove_values = list(item['key_attr'].values())
                                remove_values.remove(value)
                                for v in remove_values:
                                    new_item['title'] = new_item['title'].replace(v, "") 
                                self.items.append(new_item)
                                # i+=1
                                # if i >500:
                                #     return
                
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = item["feature"]
        title = item["title"]
        key_attr = item["key_attr"]
        label = 1
        # 生成负例
        if random.random() < 0.6:
            label = 0
            # 生成易分负例
            if random.random() < 0.3:
                sample_attr_list = self.neg_attr_dict[key_attr]["un_similar_attr"]
            # 生成难分负例
            else:
                sample_attr_list = self.neg_attr_dict[key_attr]["similar_attr"]
            new_attr = random.sample(sample_attr_list, k=1)[0]
            title = title.replace(key_attr, new_attr)
        return image, title, label

#attr id match dataset
class AttrIdMatchDataset(Dataset):
    '''generate positive and negative samples for attribute matching finetuning'''
    def __init__(self, input_filename, neg_attr_dict_file, attr_to_id_file):
        with open(neg_attr_dict_file, 'r') as f:
            self.neg_attr_dict = json.load(f)

        with open(attr_to_id_file, 'r') as f:
            self.attr_to_id = json.load(f)
        # 提取数据
        self.items = []
        i = 0
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        if item['key_attr']: # 必须有属性
                            # 生成所有离散属性
                            for key, value in item['key_attr'].items():
                                new_item = copy.deepcopy(item)
                                new_item['key'] = key
                                new_item['attr'] = value
                                # 删除title节省内存
                                del  new_item["title"]
                                self.items.append(new_item)
                                # i+=1
                                # if i >500:
                                #     return
                
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = item["feature"]
        key = item["key"]
        attr = item["attr"]
        label = 1
        # 生成负例
        if random.random() < 0.5:
            label = 0
            # 生成易分负例
            if random.random() < 0:
                sample_attr_list = self.neg_attr_dict[attr]["un_similar_attr"]
            # 生成难分负例
            else:
                sample_attr_list = self.neg_attr_dict[attr]["similar_attr"]
            attr = random.sample(sample_attr_list, k=1)[0]
        
        attr_id = self.attr_to_id[attr]

        return image, attr_id, label, key
    
class SubAttrIdMatchDataset(Dataset):
    '''generate positive and negative samples for attribute matching finetuning'''
    def __init__(self, input_filename, neg_attr_dict_file, attr_to_attrvals, key_attr):
        with open(neg_attr_dict_file, 'r') as f:
            self.neg_attr_dict = json.load(f)

        with open(attr_to_attrvals, 'r') as f:
            self.attr_to_attrvals = json.load(f)
        
        key_attr_values = self.attr_to_attrvals[key_attr]
        self.id_to_attr = {}
        self.attr_to_id = {}
        for attr_id, attr_v in enumerate(key_attr_values):
            self.attr_to_id[attr_v] = attr_id
            self.id_to_attr[attr_id] = attr_v

        # 提取数据
        self.items = []
        i = 0
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        if item['key_attr']: # 必须有属性
                            # 生成所有离散属性
                            for key, value in item['key_attr'].items():
                                # 只提取该类别
                                if key == key_attr:
                                    new_item = copy.deepcopy(item)
                                    new_item['key'] = key
                                    new_item['attr'] = value
                                    # 删除title节省内存
                                    del  new_item["title"]
                                    self.items.append(new_item)
                                    i+=1
                                    if i >500:
                                        return
                
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = item["feature"]
        key = item["key"]
        attr = item["attr"]

        label = 1
        # 生成负例
        if random.random() < 0.5:
            label = 0
            # 只生成同类负例
            sample_attr_list = self.neg_attr_dict[attr]["similar_attr"]
            attr = random.sample(sample_attr_list, k=1)[0]
        
        attr_id = self.attr_to_id[attr]

        return image, attr_id, label, key
    

# 属性替换也根据频率决定的比率进行替换
class AttrMatchProbaDataset(Dataset):
    '''generate positive and negative samples for attribute matching finetuning'''
    def __init__(self, input_filename, attr_dict_file, vocab_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        with open(vocab_dict_file, 'r') as f:
            vocab_dict = json.load(f)
        self.negative_dict, self.proba_negative_dict = self.get_negative_dict(attr_dict, vocab_dict)
        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        if item['key_attr']: # 必须有属性
                            self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
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
            if random.random() > 0.5:
                split[attr_index] = np.random.choice(self.negative_dict[attr], p=self.proba_negative_dict[attr])
                split_label[attr_index] = 0 # 标签不匹配

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

def title_cat_attrmatch_collate_fn(batch):
    images = []
    titles = []
    labels = []
    for image, title, label in batch:
        images.append(image)
        titles.append(title)
        labels.append(label)
    images = torch.tensor(images)
    titles = titles
    labels = torch.tensor(labels)
    return images, titles, labels

def attr_id_match_collate_fn(batch):
    images = []
    attr_ids = []
    labels = []
    keys = []
    for image, attr_id, label, key in batch:
        images.append(image)
        attr_ids.append(attr_id)
        labels.append(label)
        keys.append(key)
    images = torch.tensor(images)
    attr_ids = torch.tensor(attr_ids)
    labels = torch.tensor(labels)
    return images, attr_ids, labels, keys