import torch
import json
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import attr_trans_label

class JsonMultiTagsDataset(Dataset):
    def __init__(self, input_filename, attr_filename, is_train):
        logging.debug(f'Loading json data from {input_filename}.')
        self.is_train = is_train
        attr_to_label, label_to_attr = attr_trans_label(attr_filename)
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r', encoding="utf-8") as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']:
                        label = np.zeros(len(label_to_attr))
                        for key, value in item['key_attr'].items():
                            if item['match'][key] != 1: continue
                            qurey = '{}{}'.format(key,value)
                            if qurey in label_to_attr:
                                label[attr_to_label[qurey]]=1
                        item['label']=label
                        self.items.append(item)
                
        logging.debug('Done loading data.')
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = np.array(self.items[idx]['feature']).astype(np.float32)
        label = self.items[idx]['label']
        return torch.from_numpy(image), label