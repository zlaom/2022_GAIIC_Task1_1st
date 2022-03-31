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

# dataset 自监督预训练任务，没有验证集
class SplitDataset(Dataset):
    def __init__(self, input_filename, word_dict):
        # 取出所有可替换的词及出现的次数比例
        words_list = []
        proba_list = []
        for word, n in word_dict.items():
            words_list.append(word)
            proba_list.append(n)
        self.words_list = words_list 
        proba_list = np.array(proba_list)
        self.proba_list = proba_list / np.sum(proba_list)

        # 提取数据
        self.items = []
        for file in input_filename.split(','):
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if item['match']['图文']: # 训练集图文必须匹配
                        self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        split = self.items[idx]['vocab_split']
        if self.is_train:
            for i, word in enumerate(split):
                if random.random() > 0.5: # 替换
                    new_word = np.random.choice(self.words_list, p=self.proba_list)
                    split[i] = new_word
                    if new_word != word: # 存在new_word和word相同的情况
                        label = 0
                else:
                    label = 1
            return image, split, label

            

# data
# train_file = 'data/split_word/fine45000.txt,data/split_word/coarse89588.txt'
train_file = 'data/split_word/fine500_sample.txt'
word_dict_file = 'utils/data_process/base_word_dict/processed_word_dict.json'

with open(word_dict_file, 'r') as f:
    word_dict = json.load(f)
    
def collate_fn(batch):
    tensors = []
    splits = []
    labels = []
    for feature, split, label in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(labels)
    tensors = torch.stack(tensors)
    labels = torch.tensor(labels)
    return tensors, splits, labels

train_dataset = SplitDataset(train_file, word_dict)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

# model
model = BertModel()
model.cuda()

# optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# loss
loss_fn = torch.nn.BCEWithLogitsLoss()


max_acc = 0
last_path = None 

correct = 0
total = 0
            
for epoch in range(max_epoch):
    model.train()

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        images, splits, labels = batch 
        images = images.cuda()
        labels = labels.float().cuda()
        
        logits = model(images, splits).squeeze()
        
        # train acc
        if (i+1)%1 == 0:
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
        

