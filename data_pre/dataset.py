import torch
import numpy as np
import tqdm

class GaiicDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_label=None, mode='pretrain') -> None:
        super().__init__()
        self.data = data
        self.attr_label = attr_label
        self.mode = mode
        
    
    def __getitem__(self, index):
        dic = {}
        dic['label'] = self.data[index]['match']['图文'] # 图文匹配的标签
        dic['title'] = self.data[index]['title']
        dic['feature'] = self.data[index]['feature']
        return dic


    def __len__(self):
        return len(self.data)
        