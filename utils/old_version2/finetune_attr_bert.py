import os
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import transformers 
from collections import OrderedDict

from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel, FuseModelWithFusehead
from model.fusecrossmodel import FuseCrossModel, FuseCrossModelWithFusehead
from utils.lr_sched import adjust_learning_rate

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

gpus = '5'
batch_size = 128
max_epoch = 100
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 12
n_img_expand = 6
cross_layers = 1

# save_dir = 'output/split_finetune/attr/fusereplace_fusehead/0l6l1lexp6/'
save_dir = 'output/split_finetune/attr/final_bert/0l6lexp6_test/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = ''

LOAD_CKPT = True
# ckpt_file = 'output/split_pretrain/final_wordmatch/wordreplace/0l6l1lexp6/0.9218.pth'

# adjust learning rate
LR_SCHED = False
lr = 1e-5
min_lr = 5e-6
warmup_epochs = 5


train_file = 'data/equal_split_word/fine45000.txt,data/equal_split_word/attr/coarse85000.txt'
# train_file = 'data/equal_split_word/fine45000.txt'
val_file = 'data/equal_split_word/fine5000.txt,data/equal_split_word/attr/coarse4588.txt'
# train_file = 'data/equal_split_word/fine45000.txt'
# vocab_dict_file = 'dataset/vocab/vocab_dict.json'
vocab_file = 'dataset/vocab/vocab.txt'
vocab_dict_file = 'dataset/vocab/vocab_dict.json'
attr_dict_file = 'data/equal_processed_data/attr_to_attrvals.json'


# data
from dataset.keyattrmatch_dataset import AttrMatchDataset, AttrMatchProbaDataset, attrmatch_collate_fn
dataset = AttrMatchDataset
collate_fn = attrmatch_collate_fn

# train_dataset = dataset(train_file, attr_dict_file, vocab_dict_file)
# val_dataset = dataset(val_file, attr_dict_file, vocab_dict_file)

train_dataset = dataset(train_file, attr_dict_file)
val_dataset = dataset(val_file, attr_dict_file)

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

# fuse model
# split_config = BertConfig(num_hidden_layers=split_layers)
# fuse_config = BertConfig(num_hidden_layers=fuse_layers)
# model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
# if LOAD_CKPT:
#     bert = transformers.BertModel.from_pretrained('hfl/chinese-macbert-base', cache_dir='data/pretrained_model/macbert_base')
#     state_dict = bert.state_dict()
#     names = list(state_dict.keys())
#     new_dict = OrderedDict()
#     for name in names:
#         if name.startswith('encoder.'):
#             new_dict['fusebert.'+name] = state_dict[name]
#     msg = model.load_state_dict(new_dict, strict=False)
#     print(msg)
# model.cuda()

# fuse model with fuse head
split_config = BertConfig(num_hidden_layers=split_layers)
fuse_config = BertConfig(num_hidden_layers=fuse_layers)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
if LOAD_CKPT:
    bert = transformers.BertModel.from_pretrained('hfl/chinese-macbert-base', cache_dir='data/pretrained_model/macbert_base')
    state_dict = bert.state_dict()
    names = list(state_dict.keys())
    new_dict = OrderedDict()
    for name in names:
        if name.startswith('encoder.'):
            new_dict['fusebert.'+name] = state_dict[name]
    msg = model.load_state_dict(new_dict, strict=False)
    print(msg)
model.cuda()

# fuse cross model
# split_config = BertConfig(num_hidden_layers=split_layers)
# fuse_config = BertConfig(num_hidden_layers=fuse_layers)
# cross_config = BertConfig(num_hidden_layers=cross_layers)
# model = FuseCrossModel(split_config, fuse_config, cross_config, vocab_file, n_img_expand=n_img_expand)
# if LOAD_CKPT:
#     model.load_state_dict(torch.load(ckpt_file))
# model.cuda()

# fuse cross model with fuse head
# split_config = BertConfig(num_hidden_layers=split_layers)
# fuse_config = BertConfig(num_hidden_layers=fuse_layers)
# cross_config = BertConfig(num_hidden_layers=cross_layers)
# model = FuseCrossModelWithFusehead(split_config, fuse_config, cross_config, vocab_file, n_img_expand=n_img_expand)
# if LOAD_CKPT:
#     model.load_state_dict(torch.load(ckpt_file), strict=False)
# model.cuda()

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
        images, splits, labels, attr_mask = batch 
        images = images.cuda()
        logits, mask = model(images, splits, word_match=True)
        logits = logits.squeeze(2).cpu()
        
        _, W = logits.shape
        labels = labels[:, :W].float()
        attr_mask = attr_mask[:, :W].float()
        
        mask = mask.to(torch.bool)
        attr_mask = attr_mask.to(torch.bool)
        attr_mask = attr_mask[mask]
        logits = logits[mask][attr_mask]
        labels = labels[mask][attr_mask]
        
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
            
        images, splits, labels, attr_mask = batch 
        
        images = images.cuda()
        
        logits, mask = model(images, splits, word_match=True)
        logits = logits.squeeze(2)
        
        _, W = logits.shape
        labels = labels[:, :W].float().cuda()
        attr_mask = attr_mask[:, :W].float().cuda()
        
        mask = mask.to(torch.bool)
        attr_mask = attr_mask.to(torch.bool)
        attr_mask = attr_mask[mask]
        logits = logits[mask][attr_mask]
        labels = labels[mask][attr_mask]

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
        save_path = save_dir + save_name + '{:.4f}'.format(acc)+'.pth'
        last_path = save_path
        torch.save(model.state_dict(), save_path)