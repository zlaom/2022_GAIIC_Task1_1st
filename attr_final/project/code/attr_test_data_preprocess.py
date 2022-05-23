import os
from tqdm import tqdm
import json
import time

from attr_config import *

old_time = time.time()
print("开始测试数据预处理")

# 加载属性字典
attr_dict_file = os.path.join(PREPROCESS_DATA_DIR, "attr_to_attrvals.json")
with open(attr_dict_file, "r") as f:
    attr_dict = json.load(f)


# ----------------[test] 基础处理,根据query提取key_attr-------------------- #
print("preprocess test data")
rets = []
years = ["2017年", "2018年", "2019年", "2020年", "2021年", "2022年"]
with open(ORIGIN_TEST_FILE, "r") as f:
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
        for query in data["query"]:
            if query != "图文":
                flag = 0
                attr_list = attr_dict[query]
                for attr in attr_list:
                    if attr in title:
                        key_attr[query] = attr
                        flag = 1
                        break
                if flag == 0:  # 检查有没有没对应上的query
                    print(data["title"])
                    print(data["query"])

        data["key_attr"] = key_attr
        data["title"] = title
        feature = data["feature"]
        del data["feature"]
        data["feature"] = feature

        rets.append(json.dumps(data, ensure_ascii=False) + "\n")

print(len(rets))
with open(PREPROCESS_TEST_FILE, "w") as f:
    f.writelines(rets)

current_time = time.time()
print(f"测试数据预处理结束耗时：{str(current_time - old_time)}s")
