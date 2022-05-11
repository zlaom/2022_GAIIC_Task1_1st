import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FusionDataset(Dataset):
    def __init__(
        self,
        key_attr,
        origin_data,
        title_predict,
        attr_predict,
    ):

        self.origin_data = origin_data
        self.title_predict = title_predict
        self.attr_predict = attr_predict

        self.key_attr = key_attr
        self.key_attr_index = {}
        for index, attr in enumerate(key_attr):
            self.key_attr_index[attr] = index

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, idx):
        origin_data = self.origin_data[idx]
        title_predict = self.title_predict[idx]
        attr_predict = self.attr_predict[idx]

        label = np.ones(len(self.key_attr)) * -1
        predict = np.ones(len(self.key_attr)) * -1
        mask = np.zeros(len(self.key_attr))

        for key_attr, value in origin_data["key_attr"].items():
            label[self.key_attr_index[key_attr]] = value
            mask[self.key_attr_index[key_attr]] = 1

        predict[self.key_attr_index["图文"]] = title_predict["key_attr"]["图文"]

        for key_attr, value in attr_predict["key_attr"].items():
            if key_attr != "图文":
                predict[self.key_attr_index[key_attr]] = value

        return predict, label, mask


def fusion_collate_fn(batch):
    predict_list = []
    label_list = []
    mask_list = []

    for predict, label, mask in batch:
        predict_list.append(predict)
        label_list.append(label)
        mask_list.append(mask)

    predict_list = torch.tensor(predict_list)
    label_list = torch.tensor(label_list)
    mask_list = torch.tensor(mask_list)

    return predict_list, label_list, mask_list
