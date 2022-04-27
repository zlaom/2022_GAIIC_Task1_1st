import os
import torch 
import json
import random
import argparse 
import numpy as np 

from torch.utils.data import DataLoader
from tqdm import tqdm 

from utils.lr_sched import adjust_learning_rate

parser = argparse.ArgumentParser('train_attr', add_help=False)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--index', default='0', type=int)
args = parser.parse_args()   

gpus = args.gpus
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# adjust learning rate
LR_SCHED = False
lr = 1e-4
min_lr = 5e-5
warmup_epochs = 5

attr_to_attrvals = 'data/tmp_data/equal_processed_data/attr_to_attrvals.json'
with open(attr_to_attrvals, 'r') as f:
    attr_to_attrvals = json.load(f)


# 遍历训练多个模型
# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

save_dir = f'data/model_data/sub_attr_mlp_similer_100/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_name = 'attr_best_model'

# data
train_file = 'data/tmp_data/equal_processed_data/fine45000.txt,data/tmp_data/equal_processed_data/coarse85000.txt'
val_file = 'data/tmp_data/equal_processed_data/fine5000.txt,data/tmp_data/equal_processed_data/coarse4588.txt'

vocab_file = 'data/tmp_data/vocab/vocab.txt'
vocab_dict_file = 'data/tmp_data/vocab/vocab_dict.json'
neg_attr_dict_file = 'data/tmp_data/equal_processed_data/neg_attr.json'
attr_to_id = 'data/tmp_data/equal_processed_data/attr_to_id.json'
macbert_base_file = 'data/pretrain_model/macbert_base'

# dataset
from dataset.keyattrmatch_dataset import TestSubAttrIdMatchDataset, attr_id_match_collate_fn
dataset = TestSubAttrIdMatchDataset
collate_fn = attr_id_match_collate_fn

train_dataset = dataset(train_file, neg_attr_dict_file, attr_to_attrvals)
val_dataset = dataset(val_file, neg_attr_dict_file, attr_to_attrvals)

train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

# # mlp model
# from model.attr_mlp import ATTR_ID_MLP
# model = ATTR_ID_MLP(attr_num=len(key_attr_values))
# model.cuda()

# # optimizer 
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# # loss
# loss_fn = torch.nn.BCEWithLogitsLoss()

# # evaluate 
# @torch.no_grad()
# def evaluate(model, val_dataloader):
#     # 重置random种子
#     random.seed(2022)
#     model.eval()
#     correct = 0
#     total = 0
#     for batch in tqdm(val_dataloader):
#         images, attr_ids, labels, keys = batch 
#         images = images.cuda()
#         attr_ids = attr_ids.cuda()
#         logits = model(images, attr_ids)
#         logits = logits.cpu()
        
#         logits = torch.sigmoid(logits)
#         logits[logits>0.5] = 1
#         logits[logits<=0.5] = 0
        
#         correct += torch.sum(labels == logits)
#         total += len(labels)
        
#     acc = correct / total
#     return acc.item()


# max_acc = 0
# last_path = None 
# correct = 0
# total = 0
# for epoch in range(max_epoch):
#     model.train()

#     for i, batch in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         if LR_SCHED:
#             lr_now = adjust_learning_rate(optimizer, max_epoch, epoch+1, warmup_epochs, lr, min_lr)
            
#         images, attr_ids, labels, keys = batch 
#         images = images.cuda()
#         attr_ids = attr_ids.cuda()
#         labels = labels.float().cuda()
#         logits = model(images, attr_ids)
        

#         # train acc
#         if (i+1)%200 == 0:
#             train_acc = correct / total
#             correct = 0
#             total = 0
#             if LR_SCHED:
#                 print('{} Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}'.format(key_attr, epoch, max_epoch, train_acc*100, lr_now))
#             else:
#                 print('{} Epoch:[{}|{}], Acc:{:.2f}%'.format(key_attr, epoch, max_epoch, train_acc*100))
                            
#         proba = torch.sigmoid(logits.cpu())
#         proba[proba>0.5] = 1
#         proba[proba<=0.5] = 0
#         correct += torch.sum(labels.cpu() == proba)
#         total += len(labels)
#         i += 1
        
#         loss = loss_fn(logits, labels)
        
#         loss.backward()
#         optimizer.step()
        
#     acc = evaluate(model, val_dataloader)
#     print(f"epoch {epoch} eval acc: {acc}")

#     if acc > max_acc:
#         max_acc = acc
#         # if last_path:
#         #     os.remove(last_path)
#         save_path = save_dir + save_name + f'_{epoch}_{acc:.4f}.pth'
#         best_save_path = save_dir + save_name + '.pth'
#         # save_path = save_dir + save_name + '.pth'
#         last_path = save_path
#         torch.save(model.state_dict(), save_path)
#         torch.save(model.state_dict(), best_save_path)
        
#     print(f"epoch {epoch} max acc: {max_acc}")