import enum
import json
import itertools


def load_attr_dict(file):
    # 读取属性字典
    with open(file, "r") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split("="), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict


def load_attr_list(file):
    # 读取属性字典
    attr_dict = load_attr_dict(file)
    attr_list = []
    for key, value in attr_dict.items():
        for v in value:
            attr_list.append("{}{}".format(key, v))
    return attr_list


def attr_trans_label(file):
    # 读取属性字典
    label_to_attr = load_attr_list(file)
    attr_to_label = {}
    for label, attr in enumerate(label_to_attr):
        attr_to_label[attr] = label
    return attr_to_label, label_to_attr
