import os
import torch 
import json
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel

from utils.lr_sched import adjust_learning_rate

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

gpus = '5'
batch_size = 1
max_epoch = 300
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 6
n_img_expand = 6

save_dir = 'output/pretrain/title/FuseRep_New/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = '0l6lexp6_fuse0.55'

# adjust learning rate
LR_SCHED = True
lr = 4e-5
min_lr = 5e-6
warmup_epochs = 5

LOAD_CKPT = False
ckpt_file = ''

# # train_file = 'data/equal_split_word/coarse89588.txt'
# train_file = 'data/equal_split_word/title/fine40000.txt,data/equal_split_word/coarse89588.txt'
# # train_file = 'data/equal_split_word/title/fine40000.txt'
# val_file = 'data/equal_split_word/title/fine700.txt,data/equal_split_word/title/coarse1412.txt'
# # val_file = 'data/equal_split_word/title/fine9000.txt'
# # train_file = 'data/equal_split_word/fine45000.txt'
# vocab_dict_file = 'dataset/vocab/vocab_dict.json'
# vocab_file = 'dataset/vocab/vocab.txt'
# attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'

train_file = 'data/new_data/divided/title/fine40000.txt'
# train_file = 'data/new_data/divided/title/fine40000.txt,data/new_data/equal_split_word/coarse89588.txt'
val_file = 'data/new_data/divided/title/fine700.txt,data/new_data/divided/title/coarse1412.txt'
vocab_dict_file = 'data/new_data/vocab/vocab_dict.json'
vocab_file = 'data/new_data/vocab/vocab.txt'
attr_dict_file = 'data/new_data/equal_processed_data/attr_to_attrvals.json'

with open(vocab_dict_file, 'r') as f:
    vocab_dict = json.load(f)


# dataset
from dataset.clsmatch_dataset import FuseReplaceDataset, cls_collate_fn
dataset = FuseReplaceDataset
collate_fn = cls_collate_fn

# data
train_dataset = dataset(train_file, attr_dict_file, vocab_dict, is_train=True)
val_dataset = dataset(val_file, attr_dict_file, vocab_dict, is_train=False)

train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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


# model
split_config = BertConfig(num_hidden_layers=split_layers)
fuse_config = BertConfig(num_hidden_layers=fuse_layers)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
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
    for batch in tqdm(val_dataloader):
        images, splits, labels = batch 
        images = images.cuda()
        logits = model(images, splits)
        logits = logits.squeeze(1).cpu()
        
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
        images, splits, labels = batch 
        
        images = images.cuda()
        labels = labels.float().cuda()
        
        logits = model(images, splits)
        logits = logits.squeeze(1)

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
    print(acc)

    if acc > max_acc:
        max_acc = acc
        if last_path:
            os.remove(last_path)
        save_path = save_dir + save_name + '_' + '{:.4f}'.format(acc)+'.pth'
        last_path = save_path
        torch.save(model.state_dict(), save_path)