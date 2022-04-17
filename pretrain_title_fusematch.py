import os
import itertools
import torch 
import json
import numpy as np 
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import copy

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel

gpus = '1'
batch_size = 128
max_epoch = 300
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 6
n_img_expand = 2

save_dir = 'output/title_pretrain/fusematch/0l6l2exp/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = ''


train_file = 'data/equal_split_word/title/fine40000.txt,data/equal_split_word/coarse89588.txt'
# train_file = 'data/equal_split_word/title/fine40000.txt'
# val_file = 'data/equal_split_word/title/fine700.txt,data/equal_split_word/title/coarse1412.txt'
val_file = 'data/equal_split_word/title/fine9000.txt'
# train_file = 'data/equal_split_word/fine45000.txt'
vocab_dict_file = 'dataset/vocab/vocab_dict.json'
vocab_file = 'dataset/vocab/vocab.txt'
attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'


# dataset 自监督预训练任务，没有验证集
class SplitDataset(Dataset):
    def __init__(self, input_filename, vocab_dict, attr_dict_file):
        with open(attr_dict_file, 'r') as f:
            attr_dict = json.load(f)
        self.negative_dict = self.get_negative_dict(attr_dict)
        # 取出所有可替换的词及出现的次数比例
        words_list = []
        proba_list = []
        for word, n in vocab_dict.items():
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
    
    def get_negative_dict(self, attr_dict):
        negative_dict = {}
        for query, attr_list in attr_dict.items():
            for attr in attr_list:
                l = attr_list.copy()
                l.remove(attr)
                negative_dict[attr] = l
        return negative_dict
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.items[idx]['feature'])
        split = self.items[idx]['vocab_split']
        split = copy.deepcopy(split) # 要做拷贝，否则会改变self.items的值
        
        split_label = torch.ones(20)
        for i, word in enumerate(split):
            if random.random() > 0.5: # 替换
                if word in self.negative_dict:
                    split[i] = random.sample(self.negative_dict[word], 1)[0]
                    split_label[i] = 0
                else:
                    new_word = np.random.choice(self.words_list, p=self.proba_list)
                    split[i] = new_word
                    if new_word != word: # 存在new_word和word相同的情况
                        split_label[i] = 0

        return image, split, split_label
            

# data
with open(vocab_dict_file, 'r') as f:
    vocab_dict = json.load(f)
    
def collate_fn(batch):
    tensors = []
    splits = []
    labels = []
    for feature, split, split_label in batch:
        tensors.append(feature)
        splits.append(split)
        labels.append(split_label)
    tensors = torch.stack(tensors)
    labels = torch.stack(labels)
    return tensors, splits, labels

train_dataset = SplitDataset(train_file, vocab_dict, attr_dict_file)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
val_dataset = SplitDataset(val_file, vocab_dict, attr_dict_file)
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
split_config = BertConfig(num_hidden_layers=split_layers)
fuse_config = BertConfig(num_hidden_layers=fuse_layers)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
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
        images, splits, labels = batch 
        images = images.cuda()
        logits, mask = model(images, splits, word_match=True)
        logits = logits.squeeze(2).cpu()
        
        _, W = logits.shape
        labels = labels[:, :W].float()
        mask = mask.to(torch.bool)
        logits = logits[mask]
        labels = labels[mask]
        
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
        
        images, splits, labels = batch 
        
        images = images.cuda()
        
        logits, mask = model(images, splits, word_match=True)
        logits = logits.squeeze(2)
        
        _, W = logits.shape
        labels = labels[:, :W].float().cuda()
        
        mask = mask.to(torch.bool)
        logits = logits[mask]
        labels = labels[mask]

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