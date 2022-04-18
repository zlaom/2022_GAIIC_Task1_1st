import os
import torch 
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 


from model.bert.bertconfig import BertConfig
from model.fusemodel import FuseModel

gpus = '0'
batch_size = 128
max_epoch = 300
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

split_layers = 0
fuse_layers = 6
n_img_expand = 2

lr = 1e-5

save_dir = 'output/title_pretrain/word_match/0l6l2exp/'
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

with open(vocab_dict_file, 'r') as f:
    vocab_dict = json.load(f)


# dataset
from dataset.wordmatch_dataset import WordReplaceDataset, FuseReplaceDataset, word_collate_fn
dataset = WordReplaceDataset 
collate_fn = word_collate_fn

train_dataset = dataset(train_file, vocab_dict)
val_dataset = dataset(val_file, vocab_dict)
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
fuse_config = BertConfig(num_hidden_layers=fuse_layers)
model = FuseModel(split_config, fuse_config, vocab_file, n_img_expand=n_img_expand)
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