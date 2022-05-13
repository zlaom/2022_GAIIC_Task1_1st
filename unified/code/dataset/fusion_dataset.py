import torch
import numpy as np
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    def __init__(self, key_attr, origin_data):

        self.origin_data = origin_data

        self.key_attr = key_attr
        self.key_attr_index = {}
        for index, attr in enumerate(key_attr):
            self.key_attr_index[attr] = index

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, idx):
        origin_data = self.origin_data[idx]

        label = np.zeros(len(self.key_attr))
        predict = np.zeros(len(self.key_attr))
        mask = np.zeros(len(self.key_attr))

        for key_attr, value in origin_data["match"].items():
            label[self.key_attr_index[key_attr]] = int(value > 0.5)

        for key_attr, value in origin_data["pred"].items():
            mask[self.key_attr_index[key_attr]] = 1
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

    predict_list = torch.tensor(predict_list).float()
    label_list = torch.tensor(label_list).float()
    mask_list = torch.tensor(mask_list).float()

    return predict_list, label_list, mask_list
