import os
import torch
import json
import argparse
from model.attr_mlp import  CatModel, SeAttrIdMatch2
from tqdm import tqdm

parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--fold", default=10, type=int)
args = parser.parse_args()

gpus = args.gpus
batch_size = 1
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

test_type = "loss"
attr_to_id =  "data/tmp_data/unequal_processed_data/attr_to_id.json"
# model_save_dir = "/home/mw/input/zlm_attr8892/best_pos_0.45/best_pos_0.45"
model_save_dir1 = "/home/mw/input/zlm_attr8892/se_s1010_e100_b256_drop0.3_pos0.47/se_s1010_e100_b256_drop0.3_pos0.47/best"
model_save_dir2 = "/home/mw/input/zlm_attr8892/cat_s11_e60_b256_drop0.3_pos0.47/cat_s11_e60_b256_drop0.3_pos0.47/best"

print(test_type)

with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

# data
# test_file = "/home/mw/project/lhq_project/unequal_data/new_data/test_data/equal_split_word_test.txt"
test_file = "/home/mw/project/lhq_project/final_data/test_data_B/equal_split_word_test.txt"
save_dir = "/home/mw/project/result/zlm/test_B_attr"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 加载多个模型
# 生成属性to to
models = []
# SE
# for fold in range(args.fold):
# for fold in [5, 9, 4, 3, 8]:
#     model = SeAttrIdMatch2()
#     model_checkpoint_path = os.path.join(
#         model_save_dir1, f"attr_model_{test_type}_fold{fold}.pth"
#     )
#     model_checkpoint = torch.load(model_checkpoint_path)
#     model.load_state_dict(model_checkpoint)
#     model.cuda()
#     model.eval()
#     models.append(model)

# CAT
for fold in range(args.fold):
# for fold in  [5, 8, 4, 3, 2]:
    model = CatModel()
    model_checkpoint_path = os.path.join(
        model_save_dir2, f"attr_model_{test_type}_fold{fold}.pth"
    )
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint)
    model.cuda()
    model.eval()
    models.append(model)

# 遍历测试集生成结果
print(f"model len {len(models)}")
result = []
count = 0
with open(test_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        image = item["feature"]
        image = torch.tensor([image]).cuda()
        item_result = {"img_name": item["img_name"], "match": {"图文": 1}}

        # kfold结果统计
        with torch.no_grad():
            for model in models:
                for key_attr, attr_value in item["key_attr"].items():
                    attr_id = attr_to_id[attr_value]
                    attr_id = torch.tensor([attr_id]).cuda()
                    predict = model(image, attr_id)
                    predict = torch.sigmoid(predict.cpu())
                    if key_attr not in item_result["match"].keys():
                        item_result["match"][key_attr] = float(predict)
                    else:
                        item_result["match"][key_attr] += float(predict)
                    # predict = torch.sigmoid(predict.cpu()) > 0.5
                    # if key_attr not in item_result["match"].keys():
                    #     item_result["match"][key_attr] = int(predict)
                    # else:
                    #     item_result["match"][key_attr] += int(predict)

        # kfold结果判断
        for key, value in item_result["match"].items():
            if key!="图文":
                if value > len(models) / 2.0:
                    item_result["match"][key] = 1
                    count+=1
                else:
                    item_result["match"][key] = 0

        result.append(json.dumps(item_result, ensure_ascii=False) + "\n")

print(count)
with open(
    os.path.join(save_dir, f"pos0.47_cat_seed11_int_{test_type}_{len(models)}fold_results.txt"), "w", encoding="utf-8"
) as f:
    f.writelines(result)
