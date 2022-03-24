import os, sys
import re
import json
import torch
import numpy as np
import itertools
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
sys.path.insert(0, './src')


from clip.model import build_model
from clip.clip import tokenize

def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r', encoding='utf-8') as f:
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

def load_model(checkpoint_path):
    model = build_model()
    model.cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    sd = {}
    for k, v in checkpoint["state_dict"].items():
        k = k.replace('module.', '')
        sd[k] = v
    model.load_state_dict(sd)
    
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()
    
    return model.eval()

checkpoint_path = "src/training/logs/demo5/checkpoints/epoch_28.pt"
test_data = "./data/preliminary_testA.txt"
attr_dict_file = "./data/attr_to_attrvals.json"
out_file = "./result/cos_0.32_attr_0.2.txt"

# build model
model = load_model(checkpoint_path)

# test
attr_dict = load_attr_dict(attr_dict_file)
rets = []
count = 0
with open(test_data, 'r', encoding='utf-8') as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        
        data = json.loads(data)
        feature = np.array(data['feature']).astype(np.float32)
        texts = [data['title'] if a=='图文' else match_attrval(data['title'], a, attr_dict) for a in data['query']]
        features = torch.from_numpy(feature)[None, ].repeat(len(texts), 1)
        tokens = tokenize(texts)
        
        features = features.cuda()
        tokens = {k: v.cuda() for k, v in tokens.items()}
        
        with torch.no_grad():
            image_features, text_features, _ = model(features, tokens)
            similarities = (image_features*text_features).sum(dim=-1)
            similarities = similarities.cpu().tolist()
        
        # if i < 10:
        #     print(data['img_name']+':')
        #     for txt, sim in zip(texts, similarities):
        #         print(txt, sim)

        dic = {}
        if similarities[0] > 0.32:
            for key in data['query']:
                dic[key] = 1
                count += 1
        else:
            for k, v in zip(data['query'], similarities):
                if k == '图文':
                    dic[k] = 0
                else:
                    if v > 0.2:
                        dic[k] = 1
                    else:
                        dic[k] = 0

        ret = {
            "img_name": data["img_name"],
            "match": dic
        }
        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')
print(count)
with open(out_file, 'w', encoding='utf-8') as f:
    f.writelines(rets)
