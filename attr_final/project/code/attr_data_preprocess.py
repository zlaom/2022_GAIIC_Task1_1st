import os
import json
import itertools
import time

from tqdm import tqdm
from attr_config import *

old_time = time.time()
print("开始训练数据预处理")


# 创建预测处理数据存储文件夹
os.makedirs(PREPROCESS_DATA_DIR, exist_ok=True)

# 原始数据路径
attr_dict_file = os.path.join(ORIGIN_DATA_DIR, "attr_to_attrvals.json")
fine_file = os.path.join(ORIGIN_DATA_DIR, "train_fine.txt")
coarse_file = os.path.join(ORIGIN_DATA_DIR, "train_coarse.txt")

# -----------------生成新的attr_dict并保存---------------------#
# 加载原始的属性字典
def load_attr_dict(file):
    # 读取属性字典
    with open(file, "r") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split("="), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict


attr_dict = load_attr_dict(attr_dict_file)

# 特殊的几个属性替换
for query, attrs in attr_dict.items():
    attrs = attrs.copy()
    for i, attr in enumerate(attrs):
        if query == "裤门襟" and attr == "拉链":
            attr_dict[query][i] = "拉链裤"
        if query == "裤门襟" and attr == "系带":
            attr_dict[query][i] = "系带裤"
        if query == "裤门襟" and attr == "松紧":
            attr_dict[query][i] = "松紧裤"
        if query == "闭合方式" and attr == "拉链":
            attr_dict[query][i] = "拉链鞋"
        if query == "闭合方式" and attr == "系带":
            attr_dict[query][i] = "系带鞋"

# 对于搜索属性可能出错的部分，重调了顺序
for query, attr_list in attr_dict.items():
    if query == "领型":
        attr_list.remove("高领")
        attr_list.append("高领")
    if query == "衣长":
        attr_list.remove("短款")
        attr_list.append("短款")
    if query == "衣长":
        attr_list.remove("长款")
        attr_list.append("长款")
    if query == "裙长":
        attr_list.remove("短裙")
        attr_list.append("短裙")
    if query == "裙长":
        attr_list.remove("长裙")
        attr_list.append("长裙")

# 保存新的属性字典
attr_save_file = os.path.join(PREPROCESS_DATA_DIR, "attr_to_attrvals.json")
with open(attr_save_file, "w") as f:
    json.dump(attr_dict, f, ensure_ascii=False, indent=4)


# -------------[fine] 移除年份，统一大写，替换特殊属性-------------#
print("preprocess fine data")
new_fine_file = os.path.join(PREPROCESS_DATA_DIR, "fine50000.txt")
rets = []
years = ["2017年", "2018年", "2019年", "2020年", "2021年", "2022年"]

with open(fine_file, "r") as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        title = data["title"]
        key_attr = data["key_attr"]
        # 删除年份
        for year in years:
            title = title.replace(year, "")
        # 统一大写
        title = title.upper()  # 字母统一为大写
        # 属性替换
        for query, attr in key_attr.items():
            # 替换特殊属性
            if query == "裤门襟" and attr == "拉链" and "无拉链" not in title:
                key_attr[query] = "拉链裤"
                title = title.replace(attr, "拉链裤")
            if query == "裤门襟" and attr == "系带":
                key_attr[query] = "系带裤"
                title = title.replace(attr, "系带裤")
            if query == "裤门襟" and attr == "松紧":
                key_attr[query] = "松紧裤"
                title = title.replace(attr, "松紧裤")
            if query == "闭合方式" and attr == "拉链":
                key_attr[query] = "拉链鞋"
                title = title.replace(attr, "拉链鞋")
            if query == "闭合方式" and attr == "系带":
                key_attr[query] = "系带鞋"
                title = title.replace(attr, "系带鞋")
            # 一个高频词的特殊处理
            if "常规厚度" not in title and "厚度常规款" not in title and "厚度常规" in title:
                title = title.replace("厚度常规", "常规厚度")

        data["key_attr"] = key_attr
        data["title"] = title

        rets.append(json.dumps(data, ensure_ascii=False) + "\n")

with open(new_fine_file, "w") as f:
    f.writelines(rets)

# -------------[coarse] 移除年份，统一大写，替换特殊属性，提取属性-------------#
print("preprocess coarse data")
pos_coarse_file = os.path.join(PREPROCESS_DATA_DIR, "coarse89588.txt")
neg_coarse_file = os.path.join(PREPROCESS_DATA_DIR, "coarse10412.txt")

pos_rets = []
neg_rets = []
years = ["2017年", "2018年", "2019年", "2020年", "2021年", "2022年"]

with open(coarse_file, "r") as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        title = data["title"]
        key_attr = {}
        # 删除年份
        for year in years:
            title = title.replace(year, "")
        # 统一大写
        title = title.upper()  # 字母统一为大写
        # 特殊属性替换
        if "拉链" in title and "裤" in title and "无拉链" not in title:
            title = title.replace("拉链", "拉链裤")
        if "系带" in title and "裤" in title:
            title = title.replace("系带", "系带裤")
        if "松紧" in title and "裤" in title:
            title = title.replace("松紧", "松紧裤")
        if "拉链" in title and ("鞋" in title or "靴" in title):
            title = title.replace("拉链", "拉链鞋")
        if "系带" in title and ("鞋" in title or "靴" in title):
            title = title.replace("系带", "系带鞋")
        # 一个高频词的特殊处理
        if "常规厚度" not in title and "厚度常规款" not in title and "厚度常规" in title:
            title = title.replace("厚度常规", "常规厚度")

        # 属性提取
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                if attr in title:
                    key_attr[query] = attr
                    data["match"][query] = 1
                    break

        data["key_attr"] = key_attr
        data["title"] = title

        if data["match"]["图文"] == 1:
            pos_rets.append(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            neg_rets.append(json.dumps(data, ensure_ascii=False) + "\n")

print(len(pos_rets))
print(len(neg_rets))
with open(pos_coarse_file, "w") as f:
    f.writelines(pos_rets)
with open(neg_coarse_file, "w") as f:
    f.writelines(neg_rets)

current_time = time.time()
print(f"训练数据预处理结束耗时：{str(current_time - old_time)}s")
