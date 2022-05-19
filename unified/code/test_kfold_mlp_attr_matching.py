import os
import torch
import json
import argparse
from model.attr_mlp import SE_ATTR_ID_MLP, CatModel, SeAttrIdMatch2
from tqdm import tqdm

parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpus", default="1", type=str)
parser.add_argument("--fold", default=10, type=int)
args = parser.parse_args()

gpus = args.gpus
batch_size = 1
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

attr_to_attrvals = "data/tmp_data/equal_processed_data/attr_to_attrvals.json"
with open(attr_to_attrvals, "r") as f:
    attr_to_attrvals = json.load(f)

attr_to_id = "data/tmp_data/equal_processed_data/attr_to_id.json"
model_save_dir = "data/model_data/unequal_attr/final_se2_1_mlp_10fold_e80_b256_drop0.3_pos0.48/best"


with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

# 加载多个模型
# 生成属性to to
models = []
for fold in range(args.fold):
    model = SeAttrIdMatch2()
    model_checkpoint_path = os.path.join(
        model_save_dir, f"attr_model_loss_fold{fold}.pth"
    )
    # model_checkpoint_path = os.path.join(model_save_dir, f"attr_model_acc_fold{fold}.pth")
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
        image = torch.tensor([image]).cuda()
        item_result = {"img_name": item["img_name"], "match": {"图文": 0}}

        # kfold结果统计
        with torch.no_grad():
            for model in models:
                for key_attr, attr_value in item["key_attr"].items():
                    attr_id = attr_to_id[attr_value]
                    attr_id = torch.tensor([attr_id]).cuda()
                    predict = model(image, attr_id)
                    predict = torch.sigmoid(predict.cpu())[0]
                    if key_attr not in item_result["match"].keys():
                        item_result["match"][key_attr] = predict
                    else:
                        item_result["match"][key_attr] += predict
                    # predict = torch.sigmoid(predict.cpu())[0] > 0.5
                    # if key_attr not in item_result["match"].keys():
                    #     item_result["match"][key_attr] = int(predict)
                    # else:
                    #     item_result["match"][key_attr] += int(predict)

        # kfold结果判断
        for key, value in item_result["match"].items():
            if value > args.fold // 2:
                item_result["match"][key] = 1
            else:
                item_result["match"][key] = 0

        result.append(json.dumps(item_result, ensure_ascii=False) + "\n")

with open(
    os.path.join(save_dir, f"cat_{args.fold}fold_results.txt"), "w", encoding="utf-8"
) as f:
    f.writelines(result)
