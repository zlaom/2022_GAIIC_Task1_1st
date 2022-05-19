import copy
import json
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np


class AttrIdMatchDataset(Dataset):
    def __init__(self, data, attr_relation, attr_to_id, pos_rate):
        self.data = data
        self.attr_relation = attr_relation
        self.attr_to_id = attr_to_id
        self.pos_rate = pos_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["feature"]
        attr_value = item["attr_value"]
        label = 1
        soft_label = 1

        if random.random() < self.pos_rate:  # 生成正例
            equal_attr = self.attr_relation[attr_value]["equal_attr"]
            # if len(equal_attr) > 1 and random.random() < 0.1:  # 正例增强
            #     soft_label = 1
            #     attr_value = random.sample(equal_attr, 1)[0]
        else:
            label = 0
            soft_label = 0
            similar_attr_list = self.attr_relation[attr_value]["similar_attr"]
            item_similar_attr_list = random.sample(similar_attr_list, 1)[0]
            attr_value = random.sample(item_similar_attr_list, 1)[0]
            # rate = random.random()
            # if rate >= 0:  # 相似负例
            #     label = 0
            #     soft_label = 0
            #     similar_attr_list = self.attr_relation[attr_value]["similar_attr"]
            #     item_similar_attr_list = random.sample(similar_attr_list, 1)[0]
            #     attr_value = random.sample(item_similar_attr_list, 1)[0]
            # elif rate > 0.1:  # 同大类负例
            #     same_category_list = self.attr_relation[attr_value][
            #         "same_category_attr"
            #     ]
            #     if len(same_category_list) > 0:
            #         label = 0
            #         soft_label = 0.1
            #         item_same_category_list = random.sample(same_category_list, 1)[0]
            #         attr_value = random.sample(item_same_category_list, 1)[0]
            # else:  # 不同大类负例
            #     unsimilar_list = self.attr_relation[attr_value]["unsimilar_attr"]
            #     if len(unsimilar_list) > 0:
            #         label = 0
            #         soft_label = 0
            #         item_unsimilar_list = random.sample(unsimilar_list, 1)[0]
            #         attr_value = random.sample(item_unsimilar_list, 1)[0]

        attr_id = self.attr_to_id[attr_value]

        return image, attr_id, label, soft_label


class AllNegAttrIdMatchDataset(Dataset):
    def __init__(self, data, attr_relation, attr_to_id):
        self.data = []
        self.attr_relation = attr_relation
        self.attr_to_id = attr_to_id

        for item in data:
            for key_attr, origin_attr in item["key_attr"].items():
                similar_attr_list = self.attr_relation[origin_attr]["similar_attr"]
                equal_attr_list = self.attr_relation[origin_attr]["equal_attr"]
                for item_similar_attr_list in similar_attr_list:
                    new_neg_item = {}
                    new_neg_item["feature"] = item["feature"]
                    new_neg_item["attr_value"] = random.sample(
                        item_similar_attr_list, 1
                    )[0]
                    new_neg_item["label"] = 0
                    self.data.append(new_neg_item)

                    # TODO 正例增强
                    pos_attr = origin_attr
                    new_pos_item = {}
                    new_pos_item["feature"] = item["feature"]
                    new_pos_item["attr_value"] = pos_attr
                    new_pos_item["label"] = 1
                    self.data.append(new_pos_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["feature"]
        attr_value = item["attr_value"]
        label = item["label"]

        attr_id = self.attr_to_id[attr_value]

        return image, attr_id, label


def attr_id_match_collate_fn(batch):
    images = []
    attr_ids_list = []
    soft_labels = []
    labels = []
    for image, attr_ids, label, soft_label in batch:
        images.append(image)
        attr_ids_list.append(attr_ids)
        labels.append(label)
        soft_labels.append(soft_label)
    images = torch.tensor(images)
    attr_ids_list = torch.tensor(attr_ids_list)
    labels = torch.tensor(labels)
    soft_labels = torch.tensor(soft_labels)
    return images, attr_ids_list, labels, soft_labels


def all_neg_attr_id_match_collate_fn(batch):
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
