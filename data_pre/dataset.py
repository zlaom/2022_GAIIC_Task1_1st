import torch
import numpy as np
import tqdm
import json

class GaiicDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_idx=None) -> None:
        super().__init__()
        self.data = data
        new_dic = {}
        
        with open('./data/attr_to_attrvals.json', 'r', encoding='utf-8') as f:
            attr_key = json.load(f)
            i = 0
            for key, value in attr_key.items():
                new_dic[key] = i
                i += 1
        self.attr_idx = new_dic
        
    
    def __getitem__(self, index):
        dic = {}
        dic['match_label'] = self.data[index]['match']['图文'] # 图文匹配的标签
        dic['title'] = self.data[index]['title']
        dic['feature'] = self.data[index]['feature']
        
        attr_labels = np.zeros(12)
        attr_index = np.array([False for _ in range(12)])
        for key, val in self.data[index]['match'].items():
            if key == '图文':
                continue
            idx = self.attr_idx[key]
            attr_labels[idx] = val
            attr_index[idx] = True
        dic['attr_index'] = attr_index
        dic['attr_labels'] = attr_labels
        return dic


    def __len__(self):
        return len(self.data)
        