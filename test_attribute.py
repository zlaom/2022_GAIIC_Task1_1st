from cgitb import text
import re
import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert import BertModel

gpus = '1'
batch_size = 128
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

attr_dict_file = 'data/original_data/attr_to_attrvals.json'
test_file = 'data/original_data/preliminary_testA.txt'
ckpt_path = 'output/attr_match/base/0.9329.pth'
out_file = "test_pred.txt"

def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict

def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    return "{}{}".format(attr, ''.join(ret))

attr_dict = load_attr_dict(attr_dict_file)

# model
ckpt = torch.load(ckpt_path)
model = BertModel()
model.load_state_dict(ckpt)
model.cuda()


# test
model.eval()
rets = []
with open(test_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        image = data['feature']
        image = torch.tensor(image)[None, ]
        image = image.cuda()
        texts = [data['title'] if query=='图文' else match_attrval(data['title'], query, attr_dict) for query in data['query']]
        title = [texts[0]]

        # with torch.no_grad():
        #     logits = model(image, title).squeeze().cpu()
        if len(texts) > 1:
            attrs = texts[1:]
            images_for_attr = image.repeat(len(texts) - 1, 1)
            with torch.no_grad():
                attr_logits = model(images_for_attr, attrs).squeeze(1).cpu()
                attr_logits = torch.sigmoid(attr_logits).tolist()
        else:
            attr_logits = []
        title_logit = [1]
        logits = title_logit + attr_logits
        ret = {"img_name": data["img_name"],
            "match": {query: int(1 if logit>0.5 else 0) for query, logit in zip(data['query'], logits)}
        }
        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')

with open(out_file, 'w') as f:
    f.writelines(rets)


