import os
import torch 
import json
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModelSentence, FuseModel

from utils.lr_sched import adjust_learning_rate
from torch.cuda import amp 
ENABLE_AMP = True


# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

fold = seed
image_dropout = 0.3

gpus = '6'
batch_size = 256
max_epoch = 200
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 6
n_img_expand = 6

save_dir = 'output/train/attr/sentence/title_pretrain_dp0.3_sentence_real/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = '0l6lexp6'



# adjust learning rate
LR_SCHED = False
lr = 1e-5
min_lr = 5e-6
warmup_epochs = 0

LOAD_CKPT = True
ckpt_file = 'output/pretrain/title/unequal/posembed_dp0.3_bz256_epoch400/0l6lexp6_0.9271.pth'

# fine_train_file = 'data/new_data/divided/attr/10fold/divide_data/fine45000_'+str(fold)+'.txt'
# coarse_train_file = 'data/new_data/equal_split_word/coarse89588.txt'

# train_file = fine_train_file+','+coarse_train_file
# val_file = 'data/new_data/divided/attr/10fold/0.25posaug/fine5000_'+str(fold)+'.txt'

# vocab_dict_file = 'data/new_data/vocab/vocab_dict.json'
# vocab_file = 'data/new_data/vocab/vocab.txt'
# attr_dict_file = 'data/new_data/equal_processed_data/dict/attr_relation_dict.json'


# pretrain title dataset
fine_train_file = 'data/new_data/divided/title/shuffle/fine35300.txt,data/new_data/divided/title/shuffle/fine9000.txt'
coarse_train_file = 'data/new_data/equal_split_word/coarse89588.txt'
train_file = fine_train_file+','+coarse_train_file
val_file = 'data/new_data/divided/title/shuffle/fine5000_0.25posaug.txt'

vocab_dict_file = 'data/new_data/vocab/vocab_dict.json'
vocab_file = 'data/new_data/vocab/vocab.txt'
attr_dict_file = 'data/new_data/equal_processed_data/dict/attr_relation_dict.json'


with open(vocab_dict_file, 'r') as f:
    vocab_dict = json.load(f)


# dataset
from dataset.attr_unequal_dataset import SingleAttrDataset, AttrSequenceDataset, cls_collate_fn
dataset = AttrSequenceDataset
collate_fn = cls_collate_fn

# data
train_dataset = dataset(train_file, attr_dict_file, vocab_dict, is_train=True)
val_dataset = dataset(val_file, attr_dict_file, vocab_dict, is_train=False)

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

# optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# loss
loss_fn = torch.nn.BCEWithLogitsLoss()

# amp
scaler = amp.GradScaler(enabled=ENABLE_AMP)

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
        
        with amp.autocast(enabled=ENABLE_AMP):
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
        
        with amp.autocast(enabled=ENABLE_AMP):
            logits = model(images, splits)
            logits = logits.squeeze(1)
            loss = loss_fn(logits, labels)
        
        # loss scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # train acc
        if (i+1)%200 == 0:
            train_acc = correct / total
            correct = 0
            total = 0
            if LR_SCHED:
                print('Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}'.format(epoch, max_epoch, train_acc*100, lr_now))
            else:
                print('Epoch:[{}|{}], Acc:{:.2f}%'.format(epoch, max_epoch, train_acc*100))
        proba = torch.sigmoid(logits.float().cpu())
        proba[proba>0.5] = 1
        proba[proba<=0.5] = 0
        new_labels = labels.cpu().clone()
        new_labels[new_labels!=0] = 1
        correct += torch.sum(new_labels == proba)
        total += len(labels)
        i += 1
        
        
        
        # loss.backward()
        # optimizer.step()
        
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