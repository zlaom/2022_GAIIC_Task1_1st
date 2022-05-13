import json
import os
import random
import argparse
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils.lr_sched import adjust_learning_rate

from dataset.fusion_dataset import FusionDataset, fusion_collate_fn
from model.fusion_model import FusionAttrMlp


parser = argparse.ArgumentParser("fusion_title_attr", add_help=False)
parser.add_argument("--gpus", default="1", type=str)
parser.add_argument("--fold", default=5, type=int)
parser.add_argument("--fold_id", default=0, type=int)
args = parser.parse_args()

# fix the seed for reproducibility
seed = 2020
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

batch_size = 128
max_epoch = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
threshold = 0.5

# adjust learning rate
LR_SCHED = False
lr = 1e-5
min_lr = 5e-5
warmup_epochs = 5

save_dir = "data/model_data/fusion/baseline/attr_layer2_no_pos_split"
best_save_dir = os.path.join(save_dir, "best/")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(best_save_dir, exist_ok=True)

# 设置日志路径
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(save_dir, "train.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

key_attr = [
    "图文",
    "领型",
    "袖长",
    "衣长",
    "版型",
    "裙长",
    "穿着方式",
    "类别",
    "裤型",
    "裤长",
    "裤门襟",
    "闭合方式",
    "鞋帮高度",
]

# 读取预测文件
origin_data = []

for split_index in range(4):
    split_origin_data = []
    with open(f"data/tmp_data/fusion/predict/no_pos_merge_{split_index}.txt", "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            split_origin_data.append(item)
    origin_data.append(np.array(split_origin_data))
# origin_data = np.array(origin_data)


# 验证
@torch.no_grad()
def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    correct = 0
    total = 0
    loss_list = []
    for batch in tqdm(val_dataloader):
        origin_predict, label, mask = batch
        origin_predict = origin_predict.cuda()
        label = label.cuda()
        mask = mask.cuda()

        new_attr_predict = model(origin_predict, mask)
        attr_mask = mask[:, 1:]
        new_attr_predict = new_attr_predict[attr_mask == 1]

        attr_label = label[:, 1:]
        attr_label = attr_label[attr_mask == 1]

        loss = loss_fn(new_attr_predict, attr_label)
        loss_list.append(loss.mean().cpu())

        new_attr_predict[new_attr_predict > threshold] = 1
        new_attr_predict[new_attr_predict <= threshold] = 0

        correct += torch.sum(attr_label == new_attr_predict).cpu()
        total += len(attr_label)

    acc = correct / total
    return acc.item(), np.mean(loss_list)


def simple_evaluate(dataloader):
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        origin_predict, label, mask = batch

        origin_attr_predict = origin_predict[:, 1:]
        attr_mask = mask[:, 1:]
        origin_attr_predict = origin_attr_predict[attr_mask == 1]

        origin_attr_predict[origin_attr_predict > threshold] = 1
        origin_attr_predict[origin_attr_predict <= threshold] = 0

        attr_label = label[:, 1:]
        attr_label = attr_label[attr_mask == 1]
        correct += torch.sum(attr_label == origin_attr_predict).cpu()
        total += len(attr_label)

    acc = correct / total
    return acc.item()


def get_all_split_by_index(data, index):
    res = []
    for split in data:
        res.extend(split[index])
    return np.array(res)


# 训练
dataset = FusionDataset
collate_fn = fusion_collate_fn

data_index = np.arange(len(origin_data[0]))
# random.shuffle(data_index)
kf = KFold(n_splits=args.fold, shuffle=True, random_state=seed)
for fold_id, (train_index, val_index) in enumerate(kf.split(data_index)):
    if fold_id == args.fold_id:
        # 划分训练集 测试集
        all_origin_data = get_all_split_by_index(origin_data, data_index)
        train_origin_data = get_all_split_by_index(origin_data, train_index)
        val_origin_data = get_all_split_by_index(origin_data, val_index)

        all_dataset = dataset(key_attr, all_origin_data)
        train_dataset = dataset(key_attr, train_origin_data)
        val_dataset = dataset(key_attr, val_origin_data)

        all_dataloader = DataLoader(
            all_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
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

        # 初始化模型
        model = FusionAttrMlp()
        model.cuda()
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # loss
        loss_fn = nn.BCELoss()

        max_score = 0
        max_acc = 0
        min_loss = np.inf
        correct = 0
        total = 0

        for epoch in range(max_epoch):

            if epoch == 0:
                evl_acc = simple_evaluate(all_dataloader)
                logging.info(f"origin all acc: {evl_acc}")
                evl_acc = simple_evaluate(train_dataloader)
                logging.info(f"origin train acc: {evl_acc}")
                evl_acc = simple_evaluate(val_dataloader)
                logging.info(f"origin eval acc: {evl_acc}")

            model.train()
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if LR_SCHED:
                    lr_now = adjust_learning_rate(
                        optimizer, max_epoch, epoch + 1, warmup_epochs, lr, min_lr
                    )
                origin_predict, label, mask = batch
                origin_predict = origin_predict.cuda()
                label = label.cuda()
                mask = mask.cuda()

                new_attr_predict = model(origin_predict, mask)

                attr_mask = mask[:, 1:]
                new_attr_predict = new_attr_predict[attr_mask == 1]

                attr_label = label[:, 1:]
                attr_label = attr_label[attr_mask == 1]

                loss = loss_fn(new_attr_predict, attr_label)

                loss.backward()
                optimizer.step()

                new_attr_predict[new_attr_predict > threshold] = 1
                new_attr_predict[new_attr_predict <= threshold] = 0

                correct += torch.sum(attr_label == new_attr_predict).cpu()
                total += len(attr_label)

                # train acc
                if (i + 1) % 40 == 0:
                    train_acc = correct / total
                    if LR_SCHED:
                        logging.info(
                            "Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}".format(
                                epoch, max_epoch, train_acc * 100, lr_now
                            )
                        )
                    else:
                        logging.info(
                            "Epoch:[{}|{}], Acc:{:.2f}%".format(
                                epoch, max_epoch, train_acc * 100
                            )
                        )
            evl_acc, evl_loss = evaluate(model, val_dataloader, loss_fn)
            logging.info(f"eval acc: {evl_acc} loss:{evl_loss}")
            if evl_acc > max_acc:
                max_acc = evl_acc
                best_save_path = best_save_dir + f"acc_fold{args.fold_id}.pth"
                torch.save(model.state_dict(), best_save_path)

            if evl_loss < min_loss:
                min_loss = evl_loss
                best_save_path = best_save_dir + f"loss_fold{args.fold_id}.pth"
                torch.save(model.state_dict(), best_save_path)

            logging.info(f"max acc: {max_acc} min loss: {min_loss}")
