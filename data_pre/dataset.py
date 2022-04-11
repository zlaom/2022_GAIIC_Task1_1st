import torch
import numpy as np
import tqdm
import json
from random import choice

class GaiicAttrDataset(torch.utils.data.Dataset):
    def __init__(self, data, ) -> None:
        super().__init__()
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GaiicMatchDataset(torch.utils.data.Dataset):
    def __init__(self, data, path) -> None:
        super().__init__()
        self.data = data
        with open(path, 'r', encoding='utf-8') as f:
            attr_key = json.load(f)
        self.attr_key = attr_key
    
    def get_random_key(self, keys, ratio=0.7):
        list_ratio = [0.3, 0.5]
        ratio = choice(list_ratio)
        ratio = 0.5
        l = int(len(keys) * ratio)
        if l == 0:
            l = 1
        np.random.shuffle(keys)
        return keys[:l]
    
    def get_title_mask(self, title, key, val, attr_key):
        # 负样本, 随机替换title的某个属性值导致图文不匹配
        values = attr_key[key]
        key_index = 0
        for i in range(len(values)):
            if val in values[i]:
                key_index = i
                break
        new_index = np.random.randint(len(values))
        while new_index == key_index:
            new_index = np.random.randint(len(values))
        sub_val = values[new_index]
        new_sub_val = sub_val[np.random.randint(len(sub_val))]

        return title.replace(val, new_sub_val, 1)

    def get_match_value(self, key, val, attr_key):
        # 正样本属性产生
        values = attr_key[key]
        key_index = 0
        for i in range(len(values)):
            if val in values[i]:
                key_index = i
                break

        sub_values = values[key_index]
        new_index = np.random.randint(len(sub_values))
        new_sub_val = sub_values[new_index]
        return new_sub_val

    def get_dis_key_value(self, keys, pick_keys, attr_key):
        dis_idx = np.random.randint(len(keys))
        while keys[dis_idx] in pick_keys:
            dis_idx = np.random.randint(len(keys))
        values = attr_key[keys[dis_idx]]
        sub_values = values[np.random.randint(len(values))]
        return sub_values[np.random.randint(len(sub_values))]

    def __getitem__(self, index):
        new_dic = {}
        new_dic['feature'] = self.data[index]['feature']
        p = np.random.rand()
        title = self.data[index]['title']
        keys = list(self.data[index]['key_attr'].keys())
        if p > 0.5:
            # 同义替换，删
            p_2 = np.random.rand()
            if p_2 > 0.5:
                keys = self.get_random_key(keys)
                for key in keys:
                    title.replace(self.data[index]['key_attr'][key], '', 1)
                    break
            else:
                pick_key = keys[np.random.randint(len(keys))]
                pick_val = self.data[index]['key_attr'][pick_key]
                new_val = self.get_match_value(pick_key, pick_val, self.attr_key)
                title.replace(pick_val, new_val, 1)
            new_dic['all_match'] = 1
        else:
            p_2 = np.random.rand()
            # 异义替换，添加
            if p_2 > 0.5:
                keys = self.get_random_key(keys)
                for key in keys:
                    title = self.get_title_mask(title, key, self.data[index]['key_attr'][key], self.attr_key)
            else:
                new_keys = list(self.attr_key.keys())
                title = self.get_dis_key_value(new_keys, keys, self.attr_key) + title
            new_dic['all_match'] = 0
        new_dic['title'] = title

        return new_dic

    def __len__(self):
        return len(self.data)


class GaiicAttrMlpDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_idx) -> None:
        super().__init__()
        self.data = data
        self.attr_idx = attr_idx
    
    def __getitem__(self, index):
        dic = {}
        dic['attr_match'] = self.data[index]['attr_match'] # 图文匹配的标签
        dic['attr_idx'] = self.attr_idx[self.data[index]['attr']]
        dic['feature'] = self.data[index]['feature']

        return dic


    def __len__(self):
        return len(self.data)


class GaiicFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_idx=None) -> None:
        super().__init__()
        self.data = data
        
    
    def __getitem__(self, index):
        dic = {}
        dic['match_label'] = self.data[index]['match']['图文'] # 图文匹配的标签
        dic['title'] = self.data[index]['title']
        dic['feature'] = self.data[index]['feature']

        return dic


    def __len__(self):
        return len(self.data)
        