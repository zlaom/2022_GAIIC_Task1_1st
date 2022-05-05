import json
import os
import random
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.model_selection import KFold

from utils.lr_sched import adjust_learning_rate

import argparse 
parser = argparse.ArgumentParser('train_attr', add_help=False)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--fold', default=5, type=int)
parser.add_argument('--fold_id', default=0, type=int)
args = parser.parse_args()   

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

batch_size = 512
max_epoch = 200
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

split_layers = 0
fuse_layers = 12
n_img_expand = 6

save_dir = f'data/model_data/attr_simple_mlp_add_{args.fold}fold_e{max_epoch}_b{batch_size}_drop0/fold{args.fold_id}/'
best_save_dir = f'data/model_data/attr_simple_mlp_{args.fold}fold_e{max_epoch}_b{batch_size}_drop0/best/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(best_save_dir):
    os.makedirs(best_save_dir)
save_name = 'attr_model'

# adjust learning rate
LR_SCHED = False
lr = 1e-4
min_lr = 5e-5
warmup_epochs = 5

# data
input_file = [
    'data/tmp_data/equal_processed_data/coarse89588.txt',
    'data/tmp_data/equal_processed_data/fine50000.txt',
]

vocab_file = 'data/tmp_data/vocab/vocab.txt'
vocab_dict_file = 'data/tmp_data/vocab/vocab_dict.json'
neg_attr_dict_file = 'data/tmp_data/equal_processed_data/neg_attr.json'
attr_to_id = 'data/tmp_data/equal_processed_data/attr_to_id.json'
macbert_base_file = 'data/pretrain_model/macbert_base'

with open(neg_attr_dict_file, 'r') as f:
    neg_attr_dict = json.load(f)
# 加载数据
all_items = []
for file in input_file:
     with open(file, 'r') as f:
         for line in tqdm(f):
            item = json.loads(line)
            # 训练集图文必须匹配
            if item['match']['图文']: 
                # 生成所有离散属性
                for attr_key, attr_value in item['key_attr'].items():
                        new_item = {}
                        new_item["feature"] = item["feature"]
                        new_item['key'] = attr_key
                        new_item['attr'] = attr_value
                        new_item['label'] = 1
                        all_items.append(new_item)
   
                        new_item = {}
                        new_item["feature"] = item["feature"]
                        new_item['key'] = attr_key
                        new_item['label'] = 0
                        sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]
                        attr_value = random.sample(sample_attr_list, k=1)[0]
                        new_item['attr'] = attr_value
                        all_items.append(new_item)

            # if len(all_items)>2000:
            #     break
all_items = np.array(all_items)

from dataset.keyattrmatch_dataset import AttrIdMatchDataset2, attr_id_match_collate_fn
dataset = AttrIdMatchDataset2
collate_fn = attr_id_match_collate_fn
# 划分训练集 测试集
kf = KFold(n_splits=args.fold, shuffle= True, random_state=seed)
kf.get_n_splits()
for fold_id, (train_index, test_index) in enumerate(kf.split(all_items)) :
    if fold_id == args.fold_id:
        # dataset
        train_data = all_items[train_index]
        val_data = all_items[test_index]

        train_dataset = dataset(train_data, attr_to_id)
        val_dataset = dataset(val_data, attr_to_id)

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

        # fuse model
        from model.attr_mlp import ATTR_ID_MLP3
        model = ATTR_ID_MLP3()
        model.cuda()

        # optimizer 
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # loss
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # evaluate 
        @torch.no_grad()
        def evaluate(model, val_dataloader):
            # 重置random种子
            random.seed(2022)
            model.eval()
            correct = 0
            total = 0
            for batch in tqdm(val_dataloader):
                images, attr_ids, labels, _ = batch 
                images = images.cuda()
                attr_ids = attr_ids.cuda()
                logits = model(images, attr_ids)
                logits = logits.cpu()
                
                logits = torch.sigmoid(logits)
                logits[logits>0.5] = 1
                logits[logits<=0.5] = 0
                
                correct += torch.sum(labels == logits)
                total += len(labels)
                
            acc = correct / total
            return acc.item()


        max_acc = 0
        last_path = None 
        correct = 0
        total = 0
        for epoch in range(max_epoch):
            model.train()

            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if LR_SCHED:
                    lr_now = adjust_learning_rate(optimizer, max_epoch, epoch+1, warmup_epochs, lr, min_lr)
                    
                images, attr_ids, labels, _ = batch 
                images = images.cuda()
                attr_ids = attr_ids.cuda()
                labels = labels.float().cuda()
                logits = model(images, attr_ids)
                

                # train acc
                if (i+1)%200 == 0:
                    train_acc = correct / total
                    correct = 0
                    total = 0
                    if LR_SCHED:
                        print('Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}'.format(epoch, max_epoch, train_acc*100, lr_now))
                    else:
                        print('Epoch:[{}|{}], Acc:{:.2f}%'.format(epoch, max_epoch, train_acc*100))
                                    
                proba = torch.sigmoid(logits.cpu())
                proba[proba>0.5] = 1
                proba[proba<=0.5] = 0
                correct += torch.sum(labels.cpu() == proba)
                total += len(labels)
                i += 1
                
                loss = loss_fn(logits, labels)
                
                loss.backward()
                optimizer.step()
                
            acc = evaluate(model, val_dataloader)
            print(f"eval acc: {acc}")

            if acc > max_acc:
                max_acc = acc
                save_path = save_dir + save_name + f'_{epoch}_{acc:.4f}.pth'
                best_save_path = best_save_dir+save_name + f'_fold{args.fold_id}.pth'
                last_path = save_path
                torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)
                    
            print(f"max acc: {max_acc}")