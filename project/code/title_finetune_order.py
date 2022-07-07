import os
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 

from model.bert.bertconfig import BertConfig
from model.fusemodel import DesignFuseModel

from utils.lr_sched import adjust_learning_rate
import argparse 

parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--gpus', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--pretrain_seed', type=int)
parser.add_argument('--pretrain_save_dir', type=str)
args = parser.parse_args()   

# fix the seed for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

pretrain_seed = args.pretrain_seed


# configuration
gpus = args.gpus
batch_size = 64
max_epoch = 2
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

image_dropout = 0.0

split_layers = 0
fuse_layers = 6
n_img_expand = 6


# adjust learning rate
LR_SCHED = False
lr = 2e-5
min_lr = 1e-5
warmup_epochs = 0

save_dir = f'temp/tmp_data/lhq_output/title_finetune/order/seed{pretrain_seed}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = f'order_seed{pretrain_seed}_seed{seed}'

FREEZE = True
LOAD_CKPT = True
best_acc = 0.0
ckpt_dir = os.path.join(args.pretrain_save_dir, 'order')
for file in os.listdir(ckpt_dir):
    filename, filesuffix = os.path.splitext(file)
    if filesuffix == '.pth':
        _acc_ = float(filename.split('_')[2])
        if best_acc < _acc_:
            best_acc = _acc_
            ckpt_file = os.path.join(ckpt_dir, file)
print(ckpt_file)

# order
train_file = 'temp/tmp_data/lhq_data/divided/title/order/fine9000.txt,temp/tmp_data/lhq_data/divided/title/order/coarse9000.txt,temp/tmp_data/lhq_data/divided/title/order/coarse9000.txt'
val_file = 'temp/tmp_data/lhq_data/divided/title/order/fine700.txt,temp/tmp_data/lhq_data/divided/title/order/coarse1412.txt'

vocab_file = 'temp/tmp_data/lhq_data/vocab/vocab.txt' 


# dataset 
from dataset.clsmatch_dataset import ITMDataset, cls_collate_fn
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
model = DesignFuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand, word_match=True)
if LOAD_CKPT:
    model.load_state_dict(torch.load(ckpt_file))
model.cuda()

# freezing
if FREEZE:
    freeze_layers = ['splitbert']
    for name, param in model.named_parameters():
        for i in freeze_layers:
            if i in name:
                param.requires_grad = False
                break

    for name, param in model.named_parameters():
    	if param.requires_grad==False:
    		print(name,param.size())
      
# optimizer 
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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
        
        logits, word_logits, word_mask = model(images, splits)
        n_loss = loss_fn(logits, labels.float().cuda())

        loss_list.append(n_loss.mean().cpu())
        
        logits = torch.sigmoid(logits.cpu().float())
        logits[logits>0.5] = 1
        logits[logits<=0.5] = 0

        correct += torch.sum(labels == logits)
        total += len(labels)
        
    acc = correct / total
    loss_list = torch.mean(torch.tensor(loss_list))
    model.train()
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
        
        logits, word_logits, word_mask = model(images, splits)

        # train acc
        if (i+1)%20 == 0:
            train_acc = correct / total
            correct = 0
            total = 0
            print('Epoch:[{}|{}], Acc:{:.2f}%'.format(epoch, max_epoch, train_acc*100))

            # eval and save mdoel
            acc, loss = evaluate(model, val_dataloader)
            print('Acc:{:.2f}%, Loss:{:.5f}'.format(acc*100, loss))
            # if acc > max_acc:
            #     max_acc = acc
            #     if last_path:
            #         os.remove(last_path)
            #     save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc)+'.pth'
            #     last_path = save_path
            #     torch.save(model.state_dict(), save_path)
            if loss < min_loss:
                min_loss = loss
                if loss_last_path:
                    os.remove(loss_last_path)
                save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc) + '_'  + '{:.5f}'.format(loss)+'.pth'
                loss_last_path = save_path
                torch.save(model.state_dict(), save_path)
        
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
    # if acc > max_acc:
    #     max_acc = acc
    #     if last_path:
    #         os.remove(last_path)
    #     save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc)+'.pth'
    #     last_path = save_path
    #     torch.save(model.state_dict(), save_path)
    if loss < min_loss:
        min_loss = loss
        if loss_last_path:
            os.remove(loss_last_path)
        save_path = save_dir + save_name + '_'  + '{:.4f}'.format(acc) + '_'  + '{:.5f}'.format(loss)+'.pth'
        loss_last_path = save_path
        torch.save(model.state_dict(), save_path)