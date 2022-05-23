import os
import torch
import json
import argparse
import time

from model.attr_mlp import FinalCatModel
from tqdm import tqdm
from attr_config import *

old_time = time.time()
print("开始推理测试")

# 测试参数
parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--fold_num", default=10, type=int)
parser.add_argument("--test_type", default="loss", type=str)
args = parser.parse_args()

print(args.test_type)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 加载id字典
attr_to_id = f"{PREPROCESS_DATA_DIR}/attr_to_id.json"
with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

# 结果存储路径
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# 加载多个CAT模型
models = []
for fold in range(args.fold_num):
    model = FinalCatModel()
    model_checkpoint_path = os.path.join(
        ATTR_MODEL_SAVE_DIR, f"best/attr_model_{args.test_type}_fold{fold}.pth"
    )
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint)
    model.cuda()
    model.eval()
    models.append(model)

# 遍历测试集生成结果
print(f"model num {len(models)}")
result = []
count = 0
with open(PREPROCESS_TEST_FILE, "r") as f:
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

        # kfold结果判断
        for key, value in item_result["match"].items():
            if key != "图文":
                if value > len(models) / 2.0:
                    item_result["match"][key] = 1
                    count += 1
                else:
                    item_result["match"][key] = 0

        result.append(json.dumps(item_result, ensure_ascii=False) + "\n")

print(f"pos_num: {count}")

with open(
    os.path.join(RESULT_SAVE_DIR, RESULT_SAVE_NAME),
    "w",
    encoding="utf-8",
) as f:
    f.writelines(result)

current_time = time.time()
print(f"测试耗时：{str(current_time - old_time)}s")
