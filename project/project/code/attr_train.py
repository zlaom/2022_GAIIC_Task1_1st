import os
import torch 
import json
import random
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.model_selection import KFold

from model.attr_models import CatModel

from utils.lr_sched import adjust_learning_rate
import argparse 

parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--gpus', type=str)
parser.add_argument('--total_fold', type=int)
parser.add_argument('--fold_id', type=int)
args = parser.parse_args()   

GPU=args.gpus
FOLD=args.total_fold
FOLD_ID=args.fold_id


# fix the seed for reproducibility
seed = FOLD_ID
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = True


dropout = 0.3
batch_size = 256
max_epoch = 50
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

save_dir = f'temp/tmp_data/lhq_output/attr_train/fold{FOLD_ID}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = 'fold'+str(FOLD_ID)


# adjust learning rate
LR_SCHED = True
lr = 5e-4
min_lr = 5e-6
warmup_epochs = 0

LOAD_CKPT = False
ckpt_file = ''

# 文件
input_file = [
    "temp/tmp_data/lhq_data/equal_split_word/coarse89588.txt",
    "temp/tmp_data/lhq_data/equal_split_word/fine50000.txt",
]

vocab_file = 'temp/tmp_data/lhq_data/vocab/vocab.txt'
attr_relation_dict_file = 'temp/tmp_data/lhq_data/dict/attr_relation_dict.json'
attr2id_dict_file = 'temp/tmp_data/lhq_data/dict/attr_to_id.json'
with open(attr_relation_dict_file, "r") as f:
    relation_dict = json.load(f)


# 加载数据
all_items = []
for file in input_file:
    with open(file, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            if item['key_attr']: # 必须有属性
                item = json.loads(line)
                all_items.append(item)
all_items = np.array(all_items)


# 划分训练集 验证集
kf = KFold(n_splits=FOLD, shuffle=True, random_state=0)
kf.get_n_splits()
for fold_id, (train_index, val_index) in enumerate(kf.split(all_items)):
    if fold_id == FOLD_ID:
        train_data = all_items[train_index]
        raw_val_data = all_items[val_index]

val_data = []
for item in raw_val_data:
    for query, attr in item['key_attr'].items():
        key_attr = {}
        match = {}
        new_item = {}
        new_item['feature'] = item['feature']
        if random.random() < 0.5: # 替换，随机挑选一个词替换
            label = 0
            attr_list = random.sample(relation_dict[attr]['similar_attr'], 1)[0]
            if len(attr_list) == 1:
                attr = attr_list[0]
            else:
                attr = random.sample(attr_list, 1)[0]
        else: 
            label = 1
            if relation_dict[attr]['equal_attr']:
                if random.random() < 0.25: # 正例增强
                    label = 1
                    attr = random.sample(relation_dict[attr]['equal_attr'], 1)[0]

        key_attr[query] = attr
        match[query] = label
        new_item['key_attr'] = key_attr
        new_item['match'] = match
        val_data.append(new_item)
val_data = np.array(val_data)


# dataset
from dataset.attr_dataset import AttrSequenceDataset
dataset = AttrSequenceDataset
train_dataset = dataset(train_data, attr_relation_dict_file, attr2id_dict_file, is_train=True)
val_dataset = dataset(val_data, attr_relation_dict_file, attr2id_dict_file, is_train=False)

# dataloader
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )


# model
model = CatModel(dropout=dropout)
if LOAD_CKPT:
    model.load_state_dict(torch.load(ckpt_file))
model.cuda()

# optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# loss
loss_fn = torch.nn.BCEWithLogitsLoss()


# evaluate 
@torch.no_grad()
def evaluate(model, val_dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_list = []
    for batch in tqdm(val_dataloader):
        images, attr_ids, labels = batch 
        attr_ids = attr_ids.cuda()
        images = images.cuda()
        

        logits = model(images, attr_ids)
        n_loss = loss_fn(logits, labels.float().cuda())
            
        loss_list.append(n_loss.mean().cpu())
        
        logits = torch.sigmoid(logits.cpu().float())
        logits[logits>0.5] = 1
        logits[logits<=0.5] = 0
        
        correct += torch.sum(labels == logits)
        total += len(labels)
        
    acc = correct / total
    loss_list = torch.mean(torch.tensor(loss_list))
    return acc.item(), loss_list.item()


max_acc = 0
min_loss = 1000
last_path = None 
loss_last_path = None
correct = 0
total = 0
for epoch in range(max_epoch):
    model.train()

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        if LR_SCHED:
            lr_now = adjust_learning_rate(optimizer, max_epoch, epoch+1, warmup_epochs, lr, min_lr)
        images, attr_ids, labels = batch 

        attr_ids = attr_ids.cuda()
        images = images.cuda()
        labels = labels.float().cuda()
        

        logits = model(images, attr_ids)
        loss = loss_fn(logits, labels)
        
        
        # train acc
        if (i+1)%200 == 0:
            train_acc = correct / total
            correct = 0
            total = 0
            if LR_SCHED:
                print('Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}'.format(epoch, max_epoch, train_acc*100, lr_now))
            else:
                print('Epoch:[{}|{}], Acc:{:.2f}%'.format(epoch, max_epoch, train_acc*100))
        proba = torch.sigmoid(logits.float().cpu())
        proba[proba>0.5] = 1
        proba[proba<=0.5] = 0
        new_labels = labels.cpu().clone()
        new_labels[new_labels!=0] = 1
        correct += torch.sum(new_labels == proba)
        total += len(labels)
        i += 1

        loss.backward()
        optimizer.step()
        
        
    acc, loss = evaluate(model, val_dataloader)
    print('Acc:{:.2f}%, Loss:{:.5f}'.format(acc*100, loss))

    # if acc > max_acc:
    #     max_acc = acc
    #     if last_path:
    #         os.remove(last_path)
    #     save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc)+'.pth'
    #     last_path = save_path
    #     torch.save(model.state_dict(), save_path)
    
    if loss < min_loss:
        min_loss = loss
        if loss_last_path:
            os.remove(loss_last_path)
        save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc) + '_'  + '{:.5f}'.format(loss)+'.pth'
        loss_last_path = save_path
        torch.save(model.state_dict(), save_path)