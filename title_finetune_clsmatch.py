import os
import torch 
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
batch_size = 128
max_epoch = 50
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

image_dropout = 0.3

split_layers = 0
fuse_layers = 6
n_img_expand = 6


# adjust learning rate
LR_SCHED = False
lr = 1e-5
min_lr = 1e-6
warmup_epochs = 5

save_dir = 'output/finetune/title/unequal_soft0.2/dp0.3_2neg_freq/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = ''

FREEZE = False
LOAD_CKPT = True
ckpt_file = 'output/pretrain/title/unequal_day3/randomrep_soft0.2/0l6lexp6_0.9295.pth'

train_file = 'data/new_data/divided/title/shuffle/fine9000.txt,data/new_data/divided/title/shuffle/coarse9000.txt,data/new_data/divided/title/shuffle/coarse9000.txt'
val_file = 'data/new_data/divided/title/shuffle/fine700.txt,data/new_data/divided/title/shuffle/coarse1412.txt'

vocab_file = 'data/new_data/vocab/vocab.txt'
# vocab_dict_file = 'dataset/vocab/vocab_dict.json'
# attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'


# dataset 
from dataset.clsmatch_dataset import ITMDataset, ITMAugDataset, cls_collate_fn
dataset = ITMDataset
collate_fn = cls_collate_fn

# train_dataset = dataset(train_file, attr_dict_file, vocab_dict_file)
train_dataset = dataset(train_file)
val_dataset = ITMDataset(val_file)

train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
fuse_config = BertConfig(num_hidden_layers=fuse_layers, image_dropout=image_dropout)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
if LOAD_CKPT:
    model.load_state_dict(torch.load(ckpt_file))
model.cuda()

# freezing
if FREEZE:
    unfreeze_layers = ['cls_token', 'head']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in unfreeze_layers:
            if i in name:
                param.requires_grad = True
                break

    # for name, param in model.named_parameters():
    # 	if param.requires_grad:
    # 		print(name,param.size())
    
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
        images, splits, labels = batch 
        images = images.cuda()
        
        logits = model(images, splits)
        logits = logits.squeeze(1)
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
        images, splits, labels = batch 
        
        images = images.cuda()
        labels = labels.float().cuda()
        
        logits = model(images, splits)
        logits = logits.squeeze(1)

        # train acc
        if (i+1)%80 == 0:
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
        
    # eval and save mdoel
    acc, loss = evaluate(model, val_dataloader)
    print('Acc:{:.2f}%, Loss:{:.5f}'.format(acc*100, loss))
    if acc > max_acc:
        max_acc = acc
        if last_path:
            os.remove(last_path)
        save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc)+'.pth'
        last_path = save_path
        torch.save(model.state_dict(), save_path)
    if loss < min_loss:
        min_loss = loss
        if loss_last_path:
            os.remove(loss_last_path)
        save_path = save_dir + save_name + '_'  + '{:.5f}'.format(loss)+'.pth'
        loss_last_path = save_path
        torch.save(model.state_dict(), save_path)