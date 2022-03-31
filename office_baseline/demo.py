import os, sys
import re
import json
import torch
import numpy as np
import itertools
import tqdm

sys.path.insert(0, "./src")

from clip.model import build_model
from clip.clip import tokenize


def load_attr_dict(file):
    # 读取属性字典
    with open(file, "r") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split("="), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict


def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])  # 拼接
    ret = re.findall(attrvals, title)  # 找所有？
    return "{}{}".format(attr, "".join(ret))


def load_model(checkpoint_path):
    model = build_model()  # 半精度加载
    model.cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    sd = {}
    for k, v in checkpoint["state_dict"].items():
        k = k.replace("module.", "")  # 单卡替换多卡
        sd[k] = v
    model.load_state_dict(sd)

    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

    return model.eval()


checkpoint_path = "./logs/demo/checkpoints/epoch_6.pt"
test_data = "./data/test.txt"
attr_dict_file = "./data/attr_to_attrvals.json"
out_file = "test_pred.txt"

# build model
model = load_model(checkpoint_path)

# test
attr_dict = load_attr_dict(attr_dict_file)
rets = []
with open(test_data, "r") as f:
    for i, data in enumerate(tqdm.tqdm(f)):

        data = json.loads(data)
        feature = np.array(data["feature"]).astype(np.float32)
        texts = [
            data["title"] if a == "图文" else match_attrval(data["title"], a, attr_dict)
            for a in data["query"]
        ]
        features = torch.from_numpy(feature)[None,].repeat(len(texts), 1)
        tokens = tokenize(texts)

        features = features.cuda()
        tokens = {k: v.cuda() for k, v in tokens.items()}

        with torch.no_grad():
            image_features, text_features, _ = model(features, tokens)
            similarities = (image_features * text_features).sum(dim=-1)
            similarities = similarities.cpu().tolist()

        if i < 10:
            print(data["img_name"] + ":")
            for txt, sim in zip(texts, similarities):
                print(txt, sim)

        ret = {
            "img_name": data["img_name"],
            "match": {
                a: int(s > 0.4 if a == "图文" else s > 0.04)
                for a, s in zip(data["query"], similarities)
            },
        }
        rets.append(json.dumps(ret, ensure_ascii=False) + "\n")

with open(out_file, "w") as f:
    f.writelines(rets)
