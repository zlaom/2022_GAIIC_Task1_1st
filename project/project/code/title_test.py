import os
import torch 
import json
import numpy as np 
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel2Tasks

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

gpus = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 6
n_img_expand = 6

test_file = '/home/mw/project/lhq_project/final_data/test_data/equal_split_word_test.txt'
vocab_file = '/home/mw/input/lhq_data4694/necessary_files/necessary_files/vocab/vocab.txt'

out_dir = '/home/mw/project/result/lhq/title/order_0_5/remake/int/'
os.makedirs(out_dir, exist_ok=True)

model_folds = [
    '/home/mw/input/lhq_data4694/order_0_5_fold/order_0_5_fold/fold0_seed0_seed2_0.9465_0.14620.pth',
    '/home/mw/input/lhq_data4694/order_0_5_fold/order_0_5_fold/fold5_seed0_seed3_0.9493_0.13545.pth',
    '/home/mw/input/lhq_data4694/order_0_5_fold/order_0_5_fold/order_seed0_seed0_0.9484_0.13816.pth',
]
for fold_id, model_fold in enumerate(model_folds):
    fold_id = model_fold.split('/')[-1][4]
    ckpt_path = model_fold
    out_file = os.path.join(out_dir, 'fold'+str(fold_id)+'.txt')
    print(fold_id)

    # model
    split_config = BertConfig(num_hidden_layers=split_layers)
    fuse_config = BertConfig(num_hidden_layers=fuse_layers)
    model = FuseModel2Tasks(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand, word_match=True)
    model.load_state_dict(torch.load(ckpt_path))
    model.cuda()


    # test
    model.eval()
    rets = []
    with open(test_file, 'r') as f:
        for i, data in enumerate(tqdm(f)):
            data = json.loads(data)
            image = data['feature']
            image = torch.tensor(image)
            split = data['vocab_split']
            key_attr = data['key_attr']
            
            image = image[None, ].cuda()
            split = [split]

            with torch.no_grad():
                logits, word_logits, word_mask = model(image, split)
                logits = logits.cpu()
                logits = torch.sigmoid(logits)
                logits[logits>0.5] = 1
                logits[logits<=0.5] = 0

            match = {}
            # match['图文'] = logits.item()
            match['图文'] = int(logits.item())
            for query, attr in key_attr.items():
                match[query] = 1
            
            ret = {"img_name": data["img_name"],
                "match": match
            }
            rets.append(json.dumps(ret, ensure_ascii=False)+'\n')

    with open(out_file, 'w') as f:
        f.writelines(rets)


