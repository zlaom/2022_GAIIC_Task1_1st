import json
import json
import os
import random
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 

import argparse 
parser = argparse.ArgumentParser('train_attr', add_help=False)
parser.add_argument('--gpus', default='2', type=str)
args = parser.parse_args()   

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

gpus = args.gpus
batch_size = 256
max_epoch = 200
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 12
n_img_expand = 6

# save_dir = 'data/model_data/attr_mlp_no_unsimiler_200/attr_best_model_149_0.9349.pth'
save_dir = 'data/model_data/attr_mlp_no_unsimiler02/attr_best_model_92_0.9346.pth'

LOAD_CKPT = True

# data
# train_file = 'data/tmp_data/equal_processed_data/fine45000.txt,data/tmp_data/equal_processed_data/coarse85000.txt'
val_file = 'data/tmp_data/equal_processed_data/fine5000.txt,data/tmp_data/equal_processed_data/coarse4588.txt'

vocab_file = 'data/tmp_data/vocab/vocab.txt'
vocab_dict_file = 'data/tmp_data/vocab/vocab_dict.json'
neg_attr_dict_file = 'data/tmp_data/equal_processed_data/neg_attr.json'
attr_to_id = 'data/tmp_data/equal_processed_data/attr_to_id.json'
id_to_attr = 'data/tmp_data/equal_processed_data/id_to_attr.json'
macbert_base_file = 'data/pretrain_model/macbert_base'

with open(id_to_attr, 'r') as f:
    negid_to_attr_attr_dict = json.load(f)

# dataset
from dataset.keyattrmatch_dataset import AttrIdMatchDataset, attr_id_match_collate_fn
dataset = AttrIdMatchDataset
collate_fn = attr_id_match_collate_fn

# train_dataset = dataset(train_file, neg_attr_dict_file, attr_to_id)
val_dataset = dataset(val_file, neg_attr_dict_file, attr_to_id)

# train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=collate_fn,
#     )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

# fuse model
from model.attr_mlp import ATTR_ID_MLP
model = ATTR_ID_MLP()
checkpoint = torch.load(save_dir)
model.load_state_dict(checkpoint)
model.cuda()



# evaluate 
@torch.no_grad()
def evaluate(seed, model, val_dataloader):
    # 重置random种子
    random.seed(seed)
    model.eval()
    correct = 0
    total = 0

    ku_correct = 0
    xie_correct = 0
    yi_correct = 0
    bao_correct  = 0

    ku_all = 0
    xie_all = 0
    yi_all = 0
    bao_all  = 0

    ku_pos = 0
    xie_pos = 0
    yi_pos = 0
    bao_pos  = 0



    for batch in tqdm(val_dataloader):
        images, attr_ids, labels, keys = batch 
        images = images.cuda()
        attr_ids = attr_ids.cuda()
        logits = model(images, attr_ids)
        logits = logits.cpu()
        
        logits = torch.sigmoid(logits)
        logits[logits>0.5] = 1
        logits[logits<=0.5] = 0
        
        if_correct = labels == logits
        correct += torch.sum(if_correct)
        total += len(labels)

        for index, key in enumerate(keys):
            if key in ['裤型', '裤长', '裤门襟'] :
                if if_correct[index]:
                    ku_correct+=1
                if logits[index] == 1:
                    ku_pos+=1
                ku_all+=1
            elif key in ['闭合方式', '鞋帮高度'] :
                if if_correct[index]:
                    xie_correct+=1
                if logits[index] == 1:
                    xie_pos+=1
                xie_all+=1
            elif key in ['领型', '袖长', '衣长', '版型', '裙长', '穿着方式'] :
                if if_correct[index]:
                    yi_correct+=1
                if logits[index] == 1:
                    yi_pos+=1
                yi_all+=1
            else :
                if if_correct[index]:
                    bao_correct+=1
                if logits[index] == 1:
                    bao_pos+=1
                bao_all+=1

        
    acc = correct / total
    print(f"Seed {seed} Acc 鞋:{xie_correct/xie_all:.4f}, 包:{bao_correct/bao_all:.4f}, 衣:{yi_correct/yi_all:.4f}, 裤:{ku_correct/ku_all:.4f}")
    print(f"Pos Rate 鞋:{xie_pos/xie_all:.4f}, 包:{bao_pos/bao_all:.4f}, 衣:{yi_pos/yi_all:.4f}, 裤:{ku_pos/ku_all:.4f}")
    print(f"Count Num 鞋:{xie_all}, 包:{bao_all}, 衣:{yi_all}, 裤:{ku_all}")
    return acc.item()

for seed in range(5):
    evaluate(seed, model, val_dataloader)