import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert import BertModel

gpus = '2'
batch_size = 128
max_epoch = 100
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
            

# model
model = BertModel()
ckpt = torch.load('output/title_match/base/0.8287_old.pth')
model.load_state_dict(ckpt)
model.cuda()

# data
# train_file = 'data/title_trial/coarse9000.txt,data/title_trial/coarse4500.txt'
val_file = 'data/title_small/val/coarse1412.txt,data/title_small/val/coarse700_pos.txt'

val_dataset = TitleDataset(val_file, is_train=False)
val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

# evaluate 
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
print(acc)


        

