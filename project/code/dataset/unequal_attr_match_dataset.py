import copy
import json
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np

class AttrIdMatchDataset(Dataset):
    def __init__(self, data, attr_relation, attr_to_id, pos_rate):
        self.data = []
        self.attr_relation = attr_relation
        self.attr_to_id = attr_to_id
        self.pos_rate = pos_rate
        for item in data:
            for _, attr_value in item["key_attr"].items():
                new_item = {}
                new_item["feature"] = item["feature"]
                new_item["attr_value"] = attr_value
                new_item["label"] = 1
                self.data.append(new_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["feature"]
        attr_value = item["attr_value"]
        label = 1

        if random.random() < self.pos_rate:  # 生成正例
            # 正例增强
            equal_attr = self.attr_relation[attr_value]["equal_attr"]
            if len(equal_attr) > 1 and random.random() < 0.1:  
                attr_value = random.sample(equal_attr, 1)[0]
        else:
            label = 0
            similar_attr_list = self.attr_relation[attr_value]["similar_attr"]
            item_similar_attr_list = random.sample(similar_attr_list, 1)[0]
            attr_value = random.sample(item_similar_attr_list, 1)[0]

        attr_id = self.attr_to_id[attr_value]

        return image, attr_id, label

def attr_id_match_collate_fn(batch):
    images = []
    attr_ids_list = []
    labels = []
    for image, attr_ids, label in batch:
        images.append(image)
        attr_ids_list.append(attr_ids)
        labels.append(label)
    images = torch.tensor(images)
    attr_ids_list = torch.tensor(attr_ids_list)
    labels = torch.tensor(labels)
    return images, attr_ids_list, labels
