import os
import torch 
import json
import random 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

from model.bert import BertModel

gpus = '3'
batch_size = 128
max_epoch = 200
save_root = 'output/attr_match/uni_key_attr_drop0.5/'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

os.makedirs(save_root, exist_ok=True)

# 用来生成负样本的字典
with open('../data/preprocessed_data/unit_neg_sample_dict.json', 'r', encoding='utf-8') as f:
    unit_neg_sample_dict = json.load(f)

# dataset
class TitleDataset(Dataset):
    def __init__(self, input_filename, neg_sample_dict, is_train):
        self.neg_sample_dict = neg_sample_dict
        self.is_train = is_train
        self.items = []
        for file in input_filename.split(','):
            print(f'read: {file}')
            with open(file, 'r') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    if self.is_train:
                        if item['match']['图文']: # 训练集图文必须匹配
                            if item['key_attr']: # 必须有属性
                                self.items.append(item)
                    else:
                        self.items.append(item)
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = torch.tensor(item['feature'])
        if self.is_train:
            # print(list(item['key_attr'].keys()))
            query = random.sample(list(item['key_attr'].keys()), 1)[0]
            attr = item['key_attr'][query]
            label = item['match'][query]
            if random.random() > 0.5:
                attr =  random.sample(self.neg_sample_dict[f'{query}-{attr}'], 1)[0]
                label = 0
            return image, f'{query}是{attr}', label
        else:
            query = list(item['key_attr'].keys())[0]
            attr = item['key_attr'][query]
            label = item['match'][query]
            return image, f'{query}是{attr}', label
            

# data
train_file = '../data/preprocessed_data/unit_train_fine45000.txt,../data/preprocessed_data/unit_coarse_fine89588.txt'
# train_file = '../data/preprocessed_data/unit_train_fine5000.txt'
# train_file = 'data/original_data/sample/train_fine_sample.txt'
val_file = '../data/preprocessed_data/unit_attribute_val_fine.txt'
# val_file = '../data/preprocessed_data/unit_train_fine5000.txt'
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
    print(f'epoch:{epoch} val acc:{acc}')
    last_save_path = save_root+'last'.format(epoch, acc)+'.pth'
    if acc > max_acc:
        max_acc = acc
        save_path = save_root+'epoch{}-{:.4f}'.format(epoch, acc)+'.pth'
        torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), last_save_path)
        

