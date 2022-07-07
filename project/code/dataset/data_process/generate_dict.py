from copy import deepcopy
import itertools
import json
import os

from attr import attr

SAVE_DIR = 'temp/tmp_data/lhq_data'
preprocess_dir = os.path.join(SAVE_DIR, 'dict')
if not os.path.exists(preprocess_dir):
    os.makedirs(preprocess_dir)
attr_dict_file = 'input/track1_contest_4362/train/train/attr_to_attrvals.json'


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

# print(attr_dict)


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
# print(equal_attr)

attr_save_file = os.path.join(preprocess_dir, "equal_attr_dict.json")
with open(attr_save_file, "w") as f:
    json.dump(equal_attr, f, ensure_ascii=False, indent=4)

# 生成id转换字典
attr_to_id = {}
for attr_id, attr_value in enumerate(attr_list):
    attr_to_id[attr_value] = attr_id
attr_to_id_save_file = os.path.join(preprocess_dir, "attr_to_id.json")
with open(attr_to_id_save_file, "w") as f:
    json.dump(attr_to_id, f, ensure_ascii=False, indent=4)

# 生成关系字典
all_attr_list = []
for query, query_attr_list in attr_dict.items():
    for i, item_attr_list in enumerate(query_attr_list):
        for j, attr in enumerate(item_attr_list):
            all_attr_list.append(attr)


category = [
    ["领型", "袖长", "衣长", "版型", "裙长", "穿着方式"],
    ["类别"],
    ["裤型", "裤长", "裤门襟"],
    ["闭合方式", "鞋帮高度"],
]

all_category = list(attr_dict.keys())

## 生成完全不相似字典
unsimilar_attr_dic = {}
for item_category in category:
    unsimilar_attr = []
    unsimilar_category = all_category.copy()
    for item in item_category:
        unsimilar_category.remove(item)

    for item in unsimilar_category:
        unsimilar_attr.extend(attr_dict[item])
    for item in item_category:
        unsimilar_attr_dic[item] = unsimilar_attr

# print(unsimilar_attr_dic)

## 生成同大类不相似字典
same_big_category_attr_dic = {}
for item_category in category:
    for item in item_category:
        same_big_category_attr = []
        _item_category = item_category.copy()
        _item_category.remove(item)
        for _item in _item_category:
            same_big_category_attr.extend(attr_dict[_item])
        same_big_category_attr_dic[item] = same_big_category_attr

# print(same_big_category_attr_dic)

## 生成关系字典
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
            similar_attr.append(item)

        for j, attr in enumerate(item_attr_list):
            item_dic = {}
            equal_attrs_list = item_attr_list.copy()
            equal_attrs_list.remove(attr)

            item_dic["equal_attr"] = equal_attrs_list
            item_dic["similar_attr"] = similar_attr
            item_dic["same_category_attr"] = same_big_category_attr_dic[query]
            item_dic["unsimilar_attr"] = unsimilar_attr_dic[query]

            attr_relation_dic[attr] = item_dic

# print(attr_relation_dic)

attr_save_file = os.path.join(preprocess_dir, "attr_relation_dict.json")
with open(attr_save_file, "w") as f:
    json.dump(attr_relation_dic, f, ensure_ascii=False, indent=4)
