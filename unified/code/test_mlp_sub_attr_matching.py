import os
import torch 
import json
import argparse 
from model.attr_mlp import ATTR_ID_MLP2
from tqdm import tqdm 

parser = argparse.ArgumentParser('train_attr', add_help=False)
parser.add_argument('--gpus', default='0', type=str)
args = parser.parse_args()   

gpus = args.gpus
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

attr_to_attrvals = 'data/tmp_data/equal_processed_data/attr_to_attrvals.json'
with open(attr_to_attrvals, 'r') as f:
    attr_to_attrvals = json.load(f)

save_dir = f'data/model_data/sub_attr_simple_mlp_similer_100_drop0d5'
key_attr_list = ['领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度']

# 加载多个模型
# 生成属性to to
models = {}
attr_to_id_list = {}
for key_attr in key_attr_list:
    key_attr_values = attr_to_attrvals[key_attr]
    model = ATTR_ID_MLP2(attr_num=len(key_attr_values))
    model_checkpoint_path = os.path.join(save_dir, key_attr, "attr_best_model.pth")
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint)
    model.cuda()
    models[key_attr] = model

    attr_to_id = {}
    for attr_id, attr in enumerate(key_attr_values):
        attr_to_id[attr] = attr_id
    attr_to_id_list[key_attr] = attr_to_id




# data
test_file = 'data/tmp_data/equal_processed_data/test4000.txt'
save_dir = 'data/submission/sub_mlp/'

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
        
        for key_attr, attr_value in item['key_attr'].items():
            model = models[key_attr]
            attr_id = attr_to_id_list[key_attr][attr_value]
            attr_id = torch.tensor(attr_id).cuda()
            predict = model(image, attr_id)
            predict = (predict.cpu()>0.5)
            item_result["match"][key_attr] = int(predict)
        result.append(json.dumps(item_result, ensure_ascii=False)+'\n')

with open(os.path.join(save_dir,'results.txt'), 'w', encoding='utf-8') as f:
    f.writelines(result)