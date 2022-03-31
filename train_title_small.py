import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert import BertModel

gpus = '0'
batch_size = 128
max_epoch = 100
lr = 1e-5
save_dir = 'output/title_match/title_small/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = 'lr1e-5'

os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# dataset
class TitleDataset(Dataset):
    def __init__(self, input_filename, is_train):
        self.is_train = is_train
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        title = self.items[idx]['title']
        label = self.items[idx]['match']['图文']
        return image, title, label
            

# data
# train_file = 'data/title_trial/coarse9000.txt,data/title_trial/coarse4500.txt'
train_file = 'data/title_small/train/coarse9000.txt,data/title_small/train/fine9000.txt'
val_file = 'data/title_small/val/coarse1412.txt,data/title_small/val/fine700.txt'
train_dataset = TitleDataset(train_file, is_train=True)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
val_dataset = TitleDataset(val_file, is_train=False)
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

# model
model = BertModel()
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
        if (i+1)%20 == 0:
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
        save_path = save_dir + save_name + '_{:.4f}'.format(acc) + '.pth'
        last_path = save_path
        torch.save(model.state_dict(), save_path)
        

