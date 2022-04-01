import torch
import numpy as np
import tqdm
import json

class GaiicAttrDataset(torch.utils.data.Dataset):
    def __init__(self, data, ) -> None:
        super().__init__()
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]

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
        