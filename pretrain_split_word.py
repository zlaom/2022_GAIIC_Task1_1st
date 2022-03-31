import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert import BertModel

gpus = '1'
batch_size = 128
max_epoch = 100
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# 用来生成负样本的字典
def get_negative_dict(file):
    with open(file, 'r') as f:
        new_dict = {}
        for attr, attrval_list in json.load(f).items():
            attr_dict = {}
            new_dict[attr] = attr_dict
            for x in attrval_list:
                l = attrval_list.copy()
                l.remove(x)
                x = x.split('=')
                l_noequal = list(map(lambda x: x.split('='), l))
                for k in x:
                    attr_dict[k] = list(itertools.chain.from_iterable(l_noequal))
    return new_dict

attr_dict_file = "data/original_data/attr_to_attrvals.json"
negative_dict = get_negative_dict(attr_dict_file)

# 用来生成正样本的数据增强字典
def get_positive_dict(file):
    with open(file, 'r') as f:
        new_dict = {}
        for attr, attrval_list in json.load(f).items():
            attr_dict = {}
            new_dict[attr] = attr_dict
            for x in attrval_list:
                x = x.split('=')
                for k in x:
                    attr_dict[k] = x
    return new_dict
attr_dict_file = "data/original_data/attr_to_attrvals.json"
positive_dict = get_positive_dict(attr_dict_file)

# dataset
class TitleDataset(Dataset):
    def __init__(self, input_filename, negative_dict, positive_dict, is_train):
        self.negative_dict = negative_dict
        self.positive_dict = positive_dict
        self.is_train = is_train
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if self.is_train:
                        if item['match']['图文']: # 训练集图文必须匹配
                            self.items.append(item)
                    else:
                        self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        split = self.items[idx]['vocab_split']
        if self.is_train:
            if self.items[idx]['key_attr']: # 属性存在才可能进行替换
                for query, attr in self.items[idx]['key_attr'].items():
                    if random.random() > 0.5:
                        new_attr = random.sample(self.negative_dict[query][attr], 1)[0]
                        title = title.replace(attr, new_attr)
                        label = 0 # 任意一个属性负替换则标签为0
                    else:
                        new_attr = random.sample(self.positive_dict[query][attr], 1)[0]
                        title = title.replace(attr, new_attr)
            return image, title, label
        else:
            label = self.items[idx]['match']['图文']
            return image, title, label
            

# data
train_file = 'data/train/coarse89588.txt,data/train/fine45000.txt'
# train_file = 'data/original_data/sample/train_fine_sample.txt'
val_file = 'data/val/fine5000.txt,data/val/coarse10412.txt'

def collate_fn(batch):
    tensors = []
    splits = []
    for feature, split in batch:
        tensors.append(feature)
        splits.append(split)
    tensors = torch.stack(tensors)
    return tensors, splits

train_dataset = TitleDataset(train_file, negative_dict, positive_dict, is_train=True)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
val_dataset = TitleDataset(val_file, negative_dict, positive_dict, is_train=False)
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
model = BertModel()
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
        images, titles, labels = batch 
        images = images.cuda()
        logits = model(images, titles).squeeze().cpu()
        # loss = loss_fn(logits.squeeze(), labels)
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
        
        images, titles, labels = batch 
        images = images.cuda()
        labels = labels.float().cuda()
        
        logits = model(images, titles).squeeze()
        
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
        save_path = 'output/title_match/base/'+'{:.4f}'.format(acc)+'.pth'
        last_path = save_path
        torch.save(model.state_dict(), save_path)
        

