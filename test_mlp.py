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
from models.gaiic_model import ITM_ATTR_MLP, ITM_ATTR_Model, ITM_ALL_Model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Compression Reconstruction')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    args = parser.parse_args()
    yaml_path = args.cfg_file
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    test_config = config['TEST']
    data_path = test_config['DATA_PATH']
    attr_path = test_config['ATTR_PATH']
    output_path = test_config['OUT_PATH']
    os.makedirs(output_path.split('/')[0], exist_ok=True)
   
    # 属性匹配的model
    attr_model_path = test_config['ATTR_CHECKPOINT_PATH']
    attr_model = ITM_ATTR_MLP()
    attr_model.load_state_dict(torch.load(attr_model_path))
    attr_model.cuda()
    attr_model.eval()

    # 图文匹配的model
    all_model_path = test_config['ALL_CHECKPOINT_PATH']
    all_model = ITM_ALL_Model(config['MODEL']['ALL_MATCH'])
    all_model.load_state_dict(torch.load(all_model_path))
    all_model.cuda()
    all_model.eval()
    attr_dic = collections.defaultdict(int)
    i = 0
    with open(attr_path, 'r', encoding='utf-8') as f:
        key_attr = json.load(f)
        for key in key_attr:
            attr_dic[key] = i
            i += 1
    with open('./data/attr_dic.json', 'r', encoding='utf-8') as f:
        key_dic = json.load(f)

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
            new_query = [x for x in data['query'] if x != '图文']
            attr_texts = [data['key_attr'][x] for x in new_query]
            attr_idx = [key_dic[k+v] for k,v in data['key_attr'].items()]
            attr_features = torch.from_numpy(feature)[None, ].repeat(len(attr_texts), 1)
            attr_features = attr_features.cuda()
            features = torch.from_numpy(feature)
            features = features.cuda()
            features = features.unsqueeze(0)
            attr_idx = torch.from_numpy(np.array(attr_idx)).cuda()
            with torch.no_grad():
                attr_logits = attr_model(attr_features, attr_idx)
                all_logits = all_model(features, texts)
                
                attr_logits = attr_logits.cpu().tolist()
                all_logits = all_logits.cpu().tolist()
            
            dic = {}

            if all_logits[0][1] > all_logits[0][0]:
                dic['图文'] = 1
                count += 1
            else:
                dic['图文'] = 0
            
            for key, value in zip(new_query, attr_logits):
                if value[1] > value[0]:
                    dic[key] = 1
                else:
                    dic[key] = 0

            
            ret = {
                "img_name": data["img_name"],
                "match": dic
            }
            rets.append(json.dumps(ret, ensure_ascii=False)+'\n')
    print('postive num: ', count)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)