import os, sys
import json
import torch
import numpy as np
import tqdm
import argparse
import yaml

from models.fuse_model import FuseModel
from models.hero_bert.bert_config import BertConfig

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

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

    # 图文匹配的model

    
    split_config = BertConfig(num_hidden_layers=0)
    fuse_config = BertConfig(num_hidden_layers=6)
    model = FuseModel(split_config, fuse_config, 'data/split_word/vocab/vocab.txt', n_img_expand=6)
    model.load_state_dict(torch.load('./checkpoints/finetune/seed_2_loss0.2152_val_acc0.9384_.pth'))
    model = model.cuda()
    model.eval()

    rets = []
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, data in enumerate(tqdm.tqdm(f)):
            data = json.loads(data)
            feature = np.array(data['feature']).astype(np.float32)
            texts = data['title']
            new_query = [x for x in data['query'] if x != '图文']
            split = data['vocab_split']
            split = [split]
            # print(split)
            features = torch.from_numpy(feature)
            features = features.cuda()
            features = features.unsqueeze(0)
            
            with torch.no_grad():
                all_logits = model(features, split)
                all_logits = all_logits.cpu().tolist()
            
            dic = {}
            if all_logits[0][1] > all_logits[0][0]:
                dic['图文'] = 1
                count += 1
            else:
                dic['图文'] = 0
            
            for key in new_query:
                dic[key] = 1
                
            ret = {
                "img_name": data["img_name"],
                "match": dic
            }
            rets.append(json.dumps(ret, ensure_ascii=False)+'\n')
    print('postive num: ', count)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)