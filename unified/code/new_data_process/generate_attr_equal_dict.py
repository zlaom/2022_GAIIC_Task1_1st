from copy import deepcopy
import itertools
import json
import os

from attr import attr


DATA_DIR = "data/contest_data"
SAVE_DIR = "data/tmp_data/"

preprocess_dir = os.path.join(SAVE_DIR, "unequal_processed_data")
attr_dict_file = os.path.join(DATA_DIR, "attr_to_attrvals.json")


# 加载原始的属性字典
def load_attr_dict(file):
    # 读取属性字典
    with open(file, "r") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: [x.split("=")], attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict


attr_dict = load_attr_dict(attr_dict_file)


# 特殊的几个属性替换
for query, attrs_list in attr_dict.items():
    attrs_list = attrs_list.copy()
    for i, attrs_item_list in enumerate(attrs_list):
        for j, attr in enumerate(attrs_item_list):
            if query == "裤门襟" and attr == "拉链":
                attr_dict[query][i][j] = "拉链裤"
            if query == "裤门襟" and attr == "系带":
                attr_dict[query][i][j] = "系带裤"
            if query == "裤门襟" and attr == "松紧":
                attr_dict[query][i][j] = "松紧裤"
            if query == "闭合方式" and attr == "拉链":
                attr_dict[query][i][j] = "拉链鞋"
            if query == "闭合方式" and attr == "系带":
                attr_dict[query][i][j] = "系带鞋"

print(attr_dict)


# 生成相等字典
attr_set = set()
attr_list = []
equal_attr = {}
for query, attrs_list in attr_dict.items():
    for i, attrs_item_list in enumerate(attrs_list):
        for j, attr in enumerate(attrs_item_list):
            equal_attrs_list = attrs_item_list.copy()
            equal_attrs_list.remove(attr)
            equal_attr[attr] = equal_attrs_list
            attr_set.add(attr)
            attr_list.append(attr)
print(len(attr_set), len(attr_list))
print(equal_attr)

attr_save_file = os.path.join(preprocess_dir, "equal_attr_dict.json")
with open(attr_save_file, "w") as f:
    json.dump(equal_attr, f, ensure_ascii=False, indent=4)

# 生成关系字典
all_attr_list = []
for query, query_attr_list in attr_dict.items():
    for i, item_attr_list in enumerate(query_attr_list):
        for j, attr in enumerate(item_attr_list):
            all_attr_list.append(attr)


attr_relation_dic = {}
for query, query_attr_list in attr_dict.items():
    query_all_attr = []
    for i, item_attr_list in enumerate(query_attr_list):
        for j, attr in enumerate(item_attr_list):
            query_all_attr.append(attr)
    unsimilar_attr = []
    for item in all_attr_list:
        if item not in query_all_attr:
            unsimilar_attr.append(item)

    for i, item_attr_list in enumerate(query_attr_list):
        similar_attr = []
        _similar_attr = deepcopy(query_attr_list)
        _similar_attr.remove(item_attr_list)

        for item in _similar_attr:
            similar_attr.extend(item)

        for j, attr in enumerate(item_attr_list):
            item_dic = {}
            equal_attrs_list = item_attr_list.copy()
            equal_attrs_list.remove(attr)

            item_dic["equal_attr"] = equal_attrs_list
            item_dic["similar_attr"] = similar_attr
            item_dic["unsimilar_attr"] = unsimilar_attr

            attr_relation_dic[attr] = item_dic

print(attr_relation_dic)

attr_save_file = os.path.join(preprocess_dir, "attr_relation_dict.json")
with open(attr_save_file, "w") as f:
    json.dump(attr_relation_dic, f, ensure_ascii=False, indent=4)
