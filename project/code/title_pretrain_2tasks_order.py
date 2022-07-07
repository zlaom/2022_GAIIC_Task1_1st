import os
import torch 
import json
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import DesignFuseModel

from utils.lr_sched import adjust_learning_rate
from utils.logging import my_custom_logger
import argparse 

parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--gpus', type=str)
args = parser.parse_args()

seed = 0
gpus = args.gpus

image_dropout = 0.3
word_loss_scale = 2

# fix the seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

batch_size = 256
max_epoch = 400
os.environ['CUDA_VISIBLE_DEVICES'] = gpus


split_layers = 0
fuse_layers = 6
n_img_expand = 6

save_dir = f'temp/tmp_data/lhq_output/title_pretrain/order/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = f'order_seed{seed}'

logger = my_custom_logger(os.path.join(save_dir, f"training.log"))

# adjust learning rate
LR_SCHED = True
lr = 4e-5
min_lr = 5e-6
warmup_epochs = 0

LOAD_CKPT = False
ckpt_file = ''

# order
train_file = 'temp/tmp_data/lhq_data/divided/title/order/fine40000.txt,temp/tmp_data/lhq_data/equal_split_word/coarse89588.txt'
val_file = 'temp/tmp_data/lhq_data/divided/title/order/fine700.txt,temp/tmp_data/lhq_data/divided/title/order/coarse1412.txt'
# necessary files
vocab_dict_file = 'temp/tmp_data/lhq_data/vocab/vocab_dict.json'
vocab_file = 'temp/tmp_data/lhq_data/vocab/vocab.txt'
attr_dict_file = 'temp/tmp_data/lhq_data/dict/attr_relation_dict.json'

with open(vocab_dict_file, 'r') as f:
    vocab_dict = json.load(f)


# dataset
from dataset.title_unequal_2tasks_dataset import FuseReplaceDataset, cls_collate_fn
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
fuse_config = BertConfig(num_hidden_layers=fuse_layers, image_dropout=image_dropout)
model = DesignFuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand, word_match=True)
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
        images, splits, labels, word_labels = batch 
        images = images.cuda()

        logits, word_logits, word_mask = model(images, splits)
        
        logits = logits.cpu()
        logits = torch.sigmoid(logits)
        logits[logits>0.5] = 1
        logits[logits<=0.5] = 0
        
        correct += torch.sum(labels == logits)
        total += len(labels)
        
    acc = correct / total
    return acc.item()


max_acc = 0
last_path = None 
min_loss = 1000
loss_last_path = None
correct = 0
total = 0
for epoch in range(max_epoch):
    model.train()

    loss_list = []
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        if LR_SCHED:
            lr_now = adjust_learning_rate(optimizer, max_epoch, epoch+1, warmup_epochs, lr, min_lr)
        images, splits, labels, word_labels = batch 
        
        images = images.cuda()
        labels = labels.float().cuda()
        
        logits, word_logits, word_mask = model(images, splits)
        
        # word logits process
        _, W = word_logits.shape
        word_labels = word_labels[:, :W].float().cuda()
        word_mask = word_mask.to(torch.bool)
        word_logits = word_logits[word_mask]
        word_labels = word_labels[word_mask]


        # train acc
        if (i+1)%200 == 0:
            train_acc = correct / total
            correct = 0
            total = 0
            if LR_SCHED:
                logger.info('Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}'.format(epoch, max_epoch, train_acc*100, lr_now))
            else:
                logger.info('Epoch:[{}|{}], Acc:{:.2f}%'.format(epoch, max_epoch, train_acc*100))
        proba = torch.sigmoid(logits.cpu())
        proba[proba>0.5] = 1
        proba[proba<=0.5] = 0
        correct += torch.sum(labels.cpu() == proba)
        total += len(labels)
        i += 1
        
        loss = loss_fn(logits, labels) + word_loss_scale * loss_fn(word_logits, word_labels)
        loss_list.append(loss.mean().cpu())
        loss.backward()
        optimizer.step()
        
    acc = evaluate(model, val_dataloader)
    avg_loss = torch.mean(torch.tensor(loss_list)).item()
    logger.info('Acc:{:.2f}%, Loss:{:.5f}'.format(acc*100, avg_loss))
    
    if acc > max_acc:
        max_acc = acc
        if last_path:
            os.remove(last_path)
        save_path = os.path.join(save_dir, save_name + '_' + '{:.4f}'.format(acc) + '_loss'  + '{:.5f}'.format(avg_loss) + '.pth')
        last_path = save_path
        torch.save(model.state_dict(), save_path)
    
    
    # if avg_loss < min_loss:
    #     min_loss = avg_loss
    #     if loss_last_path:
    #         os.remove(loss_last_path)
    #     save_path = save_dir + save_name + '_loss_'  + '{:.5f}'.format(avg_loss)+'.pth'
    #     loss_last_path = save_path
    #     torch.save(model.state_dict(), save_path)