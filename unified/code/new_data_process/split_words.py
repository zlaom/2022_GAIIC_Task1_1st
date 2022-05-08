import os
from ltp import LTP
from tqdm import tqdm
import json
import argparse

gpus = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

SAVE_DIR = "data/tmp_data/"
ltp_path = "data/pretrain_model/ltp_base"

preprocess_dir = os.path.join(SAVE_DIR, "unequal_processed_data")
split_word_dir = os.path.join(SAVE_DIR, "unequal_split_word")
if not os.path.exists(split_word_dir):
    os.makedirs(split_word_dir)

attr_dict_file = os.path.join(preprocess_dir, "attr_to_attrvals.json")
fine_file = os.path.join(preprocess_dir, "fine50000.txt")
coarse_pos_file = os.path.join(preprocess_dir, "coarse89588.txt")
coarse_neg_file = os.path.join(preprocess_dir, "coarse10412.txt")


save_fine_file = os.path.join(split_word_dir, "fine50000.txt")
save_coarse_pos_file = os.path.join(split_word_dir, "coarse89588.txt")
save_coarse_neg_file = os.path.join(split_word_dir, "coarse10412.txt")

vocab_dir = os.path.join(SAVE_DIR, "vocab")
if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)
word_dict_save_file = os.path.join(vocab_dir, "word_dict.json")


# ----------------------------配置分词器------------------------- #
ltp = LTP(path=ltp_path)

# 取得新的属性字典并添加到词表
def get_dict(file):
    with open(file, "r") as f:
        all_attr = []
        for attr, attrval_list in json.load(f).items():
            for x in attrval_list:
                all_attr.append(x)
    return all_attr


all_attr = get_dict(attr_dict_file)

# ltp设置
extra_words = []
extra_words.append(["牛津布", "仿皮", "吸湿", "吸汗", "防滑", "抗冲击", "微弹", "加绒"])
extra_words.append(["上青", "上青色", "上青绿", "羊绒衫"])
extra_words.append(["休闲鞋", "工装鞋", "男包", "女包", "运动裤", "休闲裤", "加厚领"])
extra_words.append(["加厚", "薄款", "厚款", "短外套", "常规厚度"])
extra_words.append(["不加绒", "无扣", "无弹力", "无弹", "无拉链"])
extra_words.append(["一粒扣", "两粒扣", "暗扣", "三粒扣", "系扣"])
extra_words.append(["大红色", "大花"])
for extra in extra_words:
    all_attr = all_attr + extra

ltp.init_dict(path="user_dict.txt", max_window=6)
ltp.add_words(words=all_attr, max_window=6)


# ----------------------------fine5000 coarse89588分词-------------------------- #
# 统计训练集的词表和词频，保存分词数据
print("split words for training data:")
fine_rets = []
coarse_rets = []
word_dict = {}
print("split words for fine 50000 data...")
with open(fine_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item["title"]])
        item["title_split"] = segment[0]
        # 重置feature顺序
        feature = item["feature"]
        del item["feature"]
        item["feature"] = feature
        fine_rets.append(json.dumps(item, ensure_ascii=False) + "\n")
        # 统计词频
        for word in segment[0]:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

print("split words for coarse 89588 data...")
with open(coarse_pos_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item["title"]])
        item["title_split"] = segment[0]
        # 重置feature顺序
        feature = item["feature"]
        del item["feature"]
        item["feature"] = feature
        coarse_rets.append(json.dumps(item, ensure_ascii=False) + "\n")
        # 统计词频
        for word in segment[0]:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

# 保存词表
with open(word_dict_save_file, "w") as f:
    json.dump(word_dict, f, ensure_ascii=False)

# 保存数据
print(len(fine_rets))
print(len(coarse_rets))
with open(save_fine_file, "w") as f:
    f.writelines(fine_rets)
with open(save_coarse_pos_file, "w") as f:
    f.writelines(coarse_rets)


# ----------------------------coarse10412单纯分词-------------------------- #
# 统计训练集的词表和词频，保存分词数据
print("split words for coarse 10412 data...")
coarse_rets = []
with open(coarse_neg_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item["title"]])
        item["title_split"] = segment[0]
        # 重置feature顺序
        feature = item["feature"]
        del item["feature"]
        item["feature"] = feature
        coarse_rets.append(json.dumps(item, ensure_ascii=False) + "\n")

# 保存数据
print(len(coarse_rets))
with open(save_coarse_neg_file, "w") as f:
    f.writelines(coarse_rets)
