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
max_epoch = 200
save_root = '../data/temp_data/attribute_match/'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus


os.makedirs(save_root, exist_ok=True)

# 用来生成负样本的字典
with open('../data/equal_processed_data/attr_to_attrvals.json', 'r', encoding='utf-8') as f:
    unit_neg_sample_dict = json.load(f)

# dataset
class TitleDataset(Dataset):
    def __init__(self, input_filename, neg_sample_dict, is_train):
        self.neg_sample_dict = neg_sample_dict
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
        neg_rate = random.random()
        if self.is_train and self.items[idx]['match']['图文']==1:
            label = 1
            if self.items[idx]['key_attr']: # 属性存在才可能进行替换
                for query, attr in self.items[idx]['key_attr'].items():
                    if random.random() < 0.5: # 负例生成
                        new_attr = random.sample(self.neg_sample_dict[f'{query}-{attr}'], 1)[0]
                        title = title.replace(attr, new_attr)
                        label = 0 # 任意一个属性负替换则标签为0
            return image, title, label
        else:
            label = self.items[idx]['match']['图文']
            return image, title, label
            

# data
train_file ='../data/preprocessed_data/unit_train_fine45000.txt,../data/preprocessed_data/unit_neg_coarse_fine6412.txt,../data/preprocessed_data/unit_coarse_fine89588.txt'
# train_file ='../data/preprocessed_data/unit_train_fine.txt.00'
# train_file ='../data/preprocessed_data/unit_train_fine.txt.01'
# train_file = 'data/original_data/sample/train_fine_sample.txt'
val_file = '../data/preprocessed_data/unit_train_fine5000.txt,../data/preprocessed_data/unit_neg_coarse_fine4000.txt'
# val_file = '../data/preprocessed_data/unit_train_fine.txt.01'
train_dataset = TitleDataset(train_file, unit_neg_sample_dict, is_train=True)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
val_dataset = TitleDataset(val_file, unit_neg_sample_dict, is_train=False)
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
            print('Epoch:[{}|{}], Acc:{:.2f}%, Progress:{:.2}'.format(epoch, max_epoch, train_acc*100, i/len(train_dataloader)*100))
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
    print(f'epoch:{epoch} val acc:{acc}')
    last_save_path = save_root+'last'.format(epoch, acc)+'.pth'
    if acc > max_acc:
        max_acc = acc
        save_path = save_root+'epoch{}-{:.4f}'.format(epoch, acc)+'.pth'
        torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), last_save_path)

