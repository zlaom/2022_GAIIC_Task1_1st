import os, sys
import re
import json
from tkinter import image_names
import torch
import numpy as np
import itertools
import tqdm
import argparse
import yaml

import collections
from models.gaiic_model import GaiicModel, BLIP_Model, ITM_Model

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_dic) -> None:
        super().__init__()
        self.data = data
        self.attr_dic = attr_dic
    
    def __getitem__(self, index):
        dic = {}
        dic['img_name'] = self.data[index]['img_name']
        dic['title'] = self.data[index]['title']
        dic['feature'] = self.data[index]['feature']
        query_attr = np.zeros(12) 
        for key in self.data[index]['query']:
            if key in self.attr_dic:
                query_attr[self.attr_dic[key]] = 1

        dic['query'] = query_attr
        return dic

    def __len__(self):
        return len(self.data)


def get_match(image_names, query, cos, logits, attr_dic, cos_threshold=0.6, logits_threshold=0.5):
    dic = {}
    attr_index = {v:k for k, v in attr_dic.items()}
    if cos > cos_threshold:
        dic['图文'] = 1
    else:
        dic['图文'] = 0
    for i in range(12):
        if query[i]:
            if logits[i] > logits_threshold:
                dic[attr_index[i]] = 1
            else:
                dic[attr_index[i]] = 0
    
    ret = {
            "img_name": image_names,
            "match": dic
            }
    return ret


def get_match_attr(query, attr_dic, all_logits):

    all_logits = [x.cpu().tolist() for x in all_logits]
    new_dic = {}
    for key in query:
        index = attr_dic[key]
        single_logits = all_logits[index]
        if single_logits[0][1] > single_logits[0][0]:
            new_dic[key] = 1
        else:
            new_dic[key] = 0
    
    return new_dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Compression Reconstruction')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    args = parser.parse_args()
    yaml_path = args.cfg_file
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    test_config = config['TEST']
    data_path = test_config['DATA_PATH']
    attr_path = test_config['ATTR_PATH']
    output_path = test_config['OUT_PATH']
    os.makedirs(output_path.split('/')[0], exist_ok=True)

    model_path = test_config['CHECKPOINT_PATH']
    model = ITM_Model(config['MODEL'])

    model.load_state_dict(torch.load(model_path))
    model.cuda()
    attr_dic = collections.defaultdict(int)
    i = 0
    with open(attr_path, 'r', encoding='utf-8') as f:
        key_attr = json.load(f)
        for key in key_attr:
            attr_dic[key] = i
            i += 1

    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data_list.append(data)

    rets = []
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, data in enumerate(tqdm.tqdm(f)):
            data = json.loads(data)
            feature = np.array(data['feature']).astype(np.float32)
            texts = data['title']
            features = torch.from_numpy(feature)
            new_query = []
            for key in data['query']:
                if key == '图文':
                    continue
                new_query.append(key)
            features = features.cuda()
            features = features.unsqueeze(0)
            with torch.no_grad():
                output, attr_logits = model(features, texts)
                output = output.cpu().tolist()
            attr_match_dic = get_match_attr(new_query, attr_dic, attr_logits)
            # print(attr_match_dic)
            dic = {}
            if output[0][1] > output[0][0]:
                for key in data['query']:
                    dic[key] = 1
                count += 1
            else:
                for key in data['query']:
                    dic[key] = 0

            curr = len(attr_match_dic.keys())
            for key, val in attr_match_dic.items():
                dic[key] = val
            
            ret = {
                "img_name": data["img_name"],
                "match": dic
            }
            rets.append(json.dumps(ret, ensure_ascii=False)+'\n')
    print('postive num: ', count)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)