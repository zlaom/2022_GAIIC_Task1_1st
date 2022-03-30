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

test_file = 'data/original_data/preliminary_testA.txt'
ckpt_path = 'output/title_match/base/0.8613.pth'
out_file = "test_pred.txt"
            
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
        title = [data['title']]

        with torch.no_grad():
            logits = model(image, title).squeeze(1).cpu()
        logits = torch.sigmoid(logits).item()
        
        ret = {"img_name": data["img_name"],
            "match": {'图文': int(1 if logits>0.5 else 0)}
        }
        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')

with open(out_file, 'w') as f:
    f.writelines(rets)


