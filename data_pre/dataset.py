from ast import Assert
import torch
import numpy as np
from tqdm import tqdm
import json
from random import choice

class GaiicAttrDataset(torch.utils.data.Dataset):
    def __init__(self, data, ) -> None:
        super().__init__()
        self.data = data
        
    def __getitem__(self, index):
        new_dic = {}
        new_dic['feature'] = self.data[index]['feature']
        new_dic['title'] = self.data[index]['title']
        new_dic['all_match'] = self.data[index]['match']['图文']
        return new_dic

    def __len__(self):
        return len(self.data)


class GaiicMatchDataset(torch.utils.data.Dataset):
    def __init__(self, input_filename, attr_dict_file, is_train):
        self.is_train = is_train
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f) 
        self.negative_dict = self.get_negative_dict(attr_dict)

        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    self.items.append(item)
    
    def get_negative_dict(self, attr_dict):
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        return negative_dict

    def __getitem__(self, index):
        item = self.items[index]
        image = torch.tensor(item['feature'])
        p = np.random.rand()
        title = item['title']
        
        if self.is_train:
            if p > 0.5:
                key = np.random.choice(list(item['key_attr'].keys()))
                val = item['key_attr'][key]
                Assert(val not in self.negative_dict)
                dis_all_values = self.negative_dict[val]
                dis_val =np.random.choice(dis_all_values)
                title = title.replace(val, dis_val)
                label = 0
            else:
                label = 1
        else:
            label = item['match']['图文']
        return image, title, label

    def __len__(self):
        return len(self.items)




class GaiicAttrMlpDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_idx) -> None:
        super().__init__()
        self.data = data
        self.attr_idx = attr_idx
    
    def get_dis_value(self, key, val):
        # 负样本, 得到这个类别不匹配的属性值
        values = list(self.attr_key[key])
        new_index = np.random.randint(len(values))
        while values[key] == val:
            new_index = np.random.randint(len(values))

        dis_val = values[new_index]
        return dis_val


    def __getitem__(self, index):
        dic = {}
        
        p = np.random.rand()
        key_attrs = self.data[index]['key_attr']
        keys = key_attrs.keys()
        pick_key = np.random.choice(keys)
        val = key_attrs[pick_key]
        if p > 0.5:
            new_val = self.get_dis_value(pick_key, val)
            dic['attr_idx'] = self.attr_idx[self.data[index]['attr']]
            dic['match'] = 0
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
        