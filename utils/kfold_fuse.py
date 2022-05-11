import os 
import json 
import torch 

fold = 5

# 加载多个模型
models = []
for fold in range(fold):
    model = SE_ATTR_ID_MLP()
    model_checkpoint_path = os.path.join(save_dir, f"attr_model_loss_fold{fold}.pth")
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint)
    model.cuda()
    model.eval()
    models.append(model)

# 遍历测试集生成结果
result = []
with open(test_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        image = item["feature"]
        image = torch.tensor(image).cuda()
        item_result = {"img_name": item["img_name"], "match": {"图文": 0}}

        # kfold结果统计
        with torch.no_grad():
            for model in models:
                for key_attr, attr_value in item["key_attr"].items():
                    attr_id = attr_to_id[attr_value]
                    attr_id = torch.tensor(attr_id).cuda()
                    predict = model(image, attr_id)
                    predict = predict.cpu() > 0.5
                    if key_attr not in item_result["match"].keys():
                        item_result["match"][key_attr] = int(predict)
                    else:
                        item_result["match"][key_attr] += int(predict)

        # kfold结果判断
        for key, value in item_result["match"].items():
            if value > args.fold // 2:
                item_result["match"][key] = 1
            else:
                item_result["match"][key] = 0

        result.append(json.dumps(item_result, ensure_ascii=False) + "\n")

with open(os.path.join(save_dir, "se_5fold_results.txt"), "w", encoding="utf-8") as f:
    f.writelines(result)
