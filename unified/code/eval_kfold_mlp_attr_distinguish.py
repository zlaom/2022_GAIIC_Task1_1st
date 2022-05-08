import os
import numpy as np
import torch
import random
import json
import argparse
from model.attr_mlp import AttrIdDistinguishMLP
from tqdm import tqdm

parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--fold", default=5, type=int)
args = parser.parse_args()

gpus = args.gpus
batch_size = 1
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
similar_rate = 1

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

attr_to_attrvals = "data/tmp_data/equal_processed_data/attr_to_attrvals.json"
with open(attr_to_attrvals, "r") as f:
    attr_to_attrvals = json.load(f)

attr_to_id = "data/tmp_data/equal_processed_data/attr_to_id.json"
neg_attr_dict_file = "data/tmp_data/equal_processed_data/neg_attr.json"
save_dir = "data/model_data/attr_dis2_simple_mlp_add_5fold_e200_b512_drop0.5/best"

with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

with open(neg_attr_dict_file, "r") as f:
    neg_attr_dict = json.load(f)

# 加载多个模型
models = []
for fold in range(args.fold):
    model = AttrIdDistinguishMLP()
    model_checkpoint_path = os.path.join(save_dir, f"attr_model_loss_fold{fold}.pth")
    # model_checkpoint_path = os.path.join(save_dir, f"fold{fold}/attr_model.pth")
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint)
    model.cuda()
    model.eval()
    models.append(model)

# data
test_file = "data/tmp_data/equal_processed_data/fine5000.txt"

# 生成测试集
# all_items = []
# with open(test_file, "r") as f:
#     for line in tqdm(f):
#         item = json.loads(line)
#         # 图文必须匹配
#         if item["match"]["图文"]:
#             # 生成所有离散属性
#             for attr_key, attr_value in item["key_attr"].items():
#                 new_item = {}
#                 new_item["feature"] = item["feature"]
#                 new_item["key"] = attr_key
#                 new_item["attr"] = attr_value
#                 new_item["label"] = 1
#                 all_items.append(new_item)

#                 new_item = {}
#                 new_item["feature"] = item["feature"]
#                 new_item["key"] = attr_key
#                 new_item["label"] = 0

#                 if random.random() < similar_rate:  # 生成同类负例
#                     sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]
#                 else:  # 生成异类负例
#                     sample_attr_list = neg_attr_dict[attr_value]["un_similar_attr"]

#                 attr_value = random.sample(sample_attr_list, k=1)[0]
#                 new_item["attr"] = attr_value
#                 all_items.append(new_item)
#         if len(all_items) > 5000:
#             break

all_items = []
with open(test_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        # 图文必须匹配
        if item["match"]["图文"]:
            # 生成所有离散属性
            for attr_key, attr_value in item["key_attr"].items():
                new_item = {}
                new_item["feature"] = item["feature"]
                new_item["key"] = attr_key
                new_item["attr"] = attr_value
                new_item["label"] = 1
                all_items.append(new_item)

                # 生成所有负例
                sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]
                for attr_value in sample_attr_list:
                    new_item = {}
                    new_item["feature"] = item["feature"]
                    new_item["key"] = attr_key
                    new_item["attr"] = attr_value
                    new_item["label"] = 0
                    all_items.append(new_item)
        # if len(all_items) > 500:
        #     break

# 遍历测试集生成结果
correct = 0
all_num = 0
predict_erro_recored = []
count_error = {}
count_all = {}
count_error_rate = {}

for item in tqdm(all_items):
    all_num += 1
    image = item["feature"]
    label = item["label"]
    attr_value = item["attr"]
    # kfold结果统计
    similar_sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]

    image_list = []
    attr_ids_list = []
    for similar_attr in similar_sample_attr_list:
        attr_ids = [attr_to_id[attr_value], attr_to_id[similar_attr]]
        image_list.append(image)
        attr_ids_list.append(attr_ids)

    image_list = torch.tensor(image_list).cuda()
    attr_ids_list = torch.tensor(attr_ids_list).cuda()

    predict_list = []
    with torch.no_grad():
        for model in models:
            predict = model(image_list, attr_ids_list)
            predict_list.append(predict.cpu().numpy())

    # 求均值
    predict = np.argmax(np.mean(predict_list, axis=0), axis=-1)  # N

    # 其中1个不等于0则认为不匹配
    if np.any(predict != 0):
        predict = 0
    else:
        predict = 1

    if attr_value not in count_all:
        count_all[attr_value] = 1
    else:
        count_all[attr_value] += 1

    # 统计准确率
    if predict == label:
        correct += 1
    else:
        err_record = {"predict_list": predict_list, "item": item}
        predict_erro_recored.append(err_record)

        if attr_value not in count_error:
            count_error[attr_value] = 1
        else:
            count_error[attr_value] += 1


print(f"acc {correct/all_num}")

for key in count_error.keys():
    count_error_rate[key] = count_error[key] / count_all[key]
print(count_error)
print(count_all)
print(count_error_rate)

file = open("data/tmp_data/error_record.txt", "w")
file.write(str(predict_erro_recored))
file.close()

with open("data/tmp_data/count_error.txt", "w") as f:
    json.dump(count_error, f, ensure_ascii=False, indent=4)

with open("data/tmp_data/count_all.txt", "w") as f:
    json.dump(count_all, f, ensure_ascii=False, indent=4)

with open("data/tmp_data/count_error_rate.txt", "w") as f:
    json.dump(count_error_rate, f, ensure_ascii=False, indent=4)
