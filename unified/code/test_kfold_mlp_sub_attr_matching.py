import os
import torch 
import json
import argparse 
from model.attr_mlp import ATTR_ID_MLP
from tqdm import tqdm 

parser = argparse.ArgumentParser('train_attr', add_help=False)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--fold', default=5, type=int)
args = parser.parse_args()   

gpus = args.gpus
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

attr_to_attrvals = 'data/tmp_data/equal_processed_data/attr_to_attrvals.json'
with open(attr_to_attrvals, 'r') as f:
    attr_to_attrvals = json.load(f)

attr_to_id = 'data/tmp_data/equal_processed_data/attr_to_id.json'
save_dir = f'data/model_data/attr_mlp_5fold_e100_b512_drop0'


with open(attr_to_id, 'r') as f:
    attr_to_id = json.load(f)

# 加载多个模型
# 生成属性to to
models = []
for fold in range(args.fold):
    model = ATTR_ID_MLP()
    model_checkpoint_path = os.path.join(save_dir, f"fold{fold}/attr_model.pth")
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint)
    model.cuda()
    model.eval()
    models.append(model)




# data
test_file = 'data/tmp_data/equal_processed_data/test4000.txt'
save_dir = 'data/submission/kfold_attr_mlp/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 遍历测试集生成结果
result = []
with open(test_file, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        image = item["feature"]
        image = torch.tensor(image).cuda()
        item_result = {
            "img_name":item["img_name"],
            "match": {"图文": 0}
        }

        # kfold结果统计
        with torch.no_grad():
            for model in models:
                for key_attr, attr_value in item['key_attr'].items():
                    attr_id = attr_to_id[attr_value]
                    attr_id = torch.tensor(attr_id).cuda()
                    predict = model(image, attr_id)
                    predict = (predict.cpu()>0.5)
                    if key_attr not in item_result["match"].keys():
                        item_result["match"][key_attr] = int(predict)
                    else:
                        item_result["match"][key_attr] += int(predict)
                    
        # kfold结果判断
        for key, value in item_result["match"].items():
            if value > args.fold//2:
                item_result["match"][key] = 1
            else:
                item_result["match"][key] = 0
        
        result.append(json.dumps(item_result, ensure_ascii=False)+'\n')

with open(os.path.join(save_dir,'results.txt'), 'w', encoding='utf-8') as f:
    f.writelines(result)