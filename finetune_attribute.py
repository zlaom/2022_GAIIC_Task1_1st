import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import copy

from model.split_bert.bertconfig import BertConfig
from model.pretrain_splitbert import PretrainSplitBert

gpus = '4'
batch_size = 128
max_epoch = 100
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

save_dir = 'output/attr_finetune/base/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = ''

ckpt_file = 'output/pretrain/base/0.8771.pth'

train_file = 'data/equal_split_word/fine45000.txt,data/equal_split_word/coarse89588.txt'
val_file = 'data/equal_split_word/fine5000.txt'
# train_file = 'data/equal_split_word/fine45000.txt'
# vocab_dict_file = 'dataset/vocab/vocab_dict.json'
vocab_file = 'dataset/vocab/vocab.txt'
attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'


# dataset 自监督预训练任务，没有验证集
class SplitDataset(Dataset):
    def __init__(self, input_filename, attr_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.negative_dict = self.get_negative_dict(attr_dict)
        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        if item['key_attr']: # 必须有属性
                            self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def get_negative_dict(self, attr_dict):
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            negative_dict[query] = {}
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[query][attr] = l
        return negative_dict
        
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        split = item['vocab_split']
        split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
        key_attr = item['key_attr']
        
        split_label = torch.ones(20)
        attr_mask = torch.zeros(20)
        for query, attr in key_attr.items():
            attr_index = split.index(attr) # 先找到属性的位置
            attr_mask[attr_index] = 1
            if random.random() > 0.5:
                new_attr = random.sample(self.negative_dict[query][attr], 1)[0]
                split[attr_index] = new_attr
                split_label[attr_index] = 0 # 标签不匹配

        return image, split, split_label, attr_mask

            

# data
def collate_fn(batch):
    tensors = []
    splits = []
    labels = []
    masks = []
    for feature, split, split_label, attr_mask in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(split_label)
        masks.append(attr_mask)
    tensors = torch.stack(tensors)
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    return tensors, splits, labels, masks

train_dataset = SplitDataset(train_file, attr_dict_file)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
val_dataset = SplitDataset(val_file, attr_dict_file)
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

# model
config = BertConfig()
model = PretrainSplitBert(config, vocab_file)
# model.load_state_dict(torch.load(ckpt_file))
model.cuda()

# optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# loss
loss_fn = torch.nn.BCEWithLogitsLoss()

# evaluate 
@torch.no_grad()
def evaluate(model, val_dataloader):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(val_dataloader):
        images, splits, labels, attr_mask = batch 
        images = images.cuda()
        logits, mask = model(images, splits)
        logits = logits.squeeze(2).cpu()
        
        _, W = logits.shape
        labels = labels[:, :W].float()
        attr_mask = attr_mask[:, :W].float()
        
        mask = mask.to(torch.bool)
        attr_mask = attr_mask.to(torch.bool)
        attr_mask = attr_mask[mask]
        logits = logits[mask][attr_mask]
        labels = labels[mask][attr_mask]
        
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
        
        images, splits, labels, attr_mask = batch 
        
        images = images.cuda()
        
        logits, mask = model(images, splits)
        logits = logits.squeeze(2)
        
        _, W = logits.shape
        labels = labels[:, :W].float().cuda()
        attr_mask = attr_mask[:, :W].float().cuda()
        
        mask = mask.to(torch.bool)
        attr_mask = attr_mask.to(torch.bool)
        attr_mask = attr_mask[mask]
        logits = logits[mask][attr_mask]
        labels = labels[mask][attr_mask]

        # train acc
        if (i+1)%200 == 0:
            train_acc = correct / total
            correct = 0
            total = 0
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
    print(acc)

    if acc > max_acc:
        max_acc = acc
        if last_path:
            os.remove(last_path)
        save_path = save_dir + save_name + '{:.4f}'.format(acc)+'.pth'
        last_path = save_path
        torch.save(model.state_dict(), save_path)