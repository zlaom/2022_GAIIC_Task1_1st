import os
import numpy as np
import torch
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

attr_to_attrvals = "data/tmp_data/equal_processed_data/attr_to_attrvals.json"
with open(attr_to_attrvals, "r") as f:
    attr_to_attrvals = json.load(f)

attr_to_id = "data/tmp_data/equal_processed_data/attr_to_id.json"
neg_attr_dict_file = "data/tmp_data/equal_processed_data/neg_attr.json"
save_dir = "data/model_data/attr_dis_simple_mlp_add_5fold_e200_b512_drop0.5/best"

with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

with open(neg_attr_dict_file, "r") as f:
    neg_attr_dict = json.load(f)

# 加载多个模型
# 生成属性to to
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
test_file = "data/tmp_data/equal_processed_data/test4000.txt"
save_dir = "data/submission/kfold_attr_mlp/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 遍历测试集生成结果
result = []
with open(test_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        image = item["feature"]
        # image = torch.tensor([image]).cuda()
        item_result = {
            "img_name": item["img_name"],
            "match": {"图文": 0},
        }

        # kfold结果统计
        for key_attr, attr_value in item["key_attr"].items():
            item_result["match"][key_attr] = 1
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
                item_result["match"][key_attr] = 0

        result.append(json.dumps(item_result, ensure_ascii=False) + "\n")

with open(os.path.join(save_dir, "dis_5fold_results.txt"), "w", encoding="utf-8") as f:
    f.writelines(result)
