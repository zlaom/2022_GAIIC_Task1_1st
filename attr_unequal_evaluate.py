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

gpus = '3'
batch_size = 128
max_epoch = 100
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 6
n_img_expand = 6

THRESH = 0.6

# adjust learning rate
LR_SCHED = False
lr = 1e-5
min_lr = 5e-6
warmup_epochs = 5

LOAD_CKPT = True
ckpt_file = 'output/train/attr/unequal/baseline/0l6lexp6_0.9456.pth'


# train_file = 'data/new_data/divided/attr/fine45000.txt'
train_file = 'data/new_data/divided/attr/fine45000.txt,data/new_data/equal_split_word/coarse89588.txt'
val_file = 'data/new_data/divided/attr/fine5000.txt'

vocab_dict_file = 'data/new_data/vocab/vocab_dict.json'
vocab_file = 'data/new_data/vocab/vocab.txt'
attr_dict_file = 'data/new_data/equal_processed_data/attr_relation_dict.json'

with open(vocab_dict_file, 'r') as f:
    vocab_dict = json.load(f)


# dataset
from dataset.attr_unequal_dataset import SingleAttrDataset, FuseReplaceDataset, cls_collate_fn
dataset = SingleAttrDataset
collate_fn = cls_collate_fn

# data
val_dataset = dataset(val_file, attr_dict_file, vocab_dict, is_train=False)

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

# loss
loss_fn = torch.nn.BCEWithLogitsLoss()

# evaluate 
@torch.no_grad()
def evaluate(model, val_dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for batch in tqdm(val_dataloader):
        images, splits, labels = batch 
        images = images.cuda()
        logits = model(images, splits)
        logits = logits.squeeze(1).cpu()
        
        total_loss += loss_fn(logits, labels.float())
        
        logits = torch.sigmoid(logits)
        logits[logits>THRESH] = 1
        logits[logits<=THRESH] = 0
        
        correct += torch.sum(labels == logits)
        total += len(labels)
        
    acc = correct / total
    loss = total_loss / total
    return acc.item(), loss.item()

acc, loss = evaluate(model, val_dataloader)
print('Acc:{:.2f}%, Loss:{:.5f}'.format(acc*100, loss))
