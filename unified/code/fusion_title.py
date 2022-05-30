import copy
import json
from operator import index
import os
import random
import argparse
import logging
import sys
from cv2 import split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils.lr_sched import adjust_learning_rate

from dataset.fusion_dataset import FusionDataset, fusion_collate_fn
from model.fusion_model import FusionTitleMlp, FusionTitleMlp2


parser = argparse.ArgumentParser("fusion_title_attr", add_help=False)
parser.add_argument("--gpus", default="0", type=str)
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

save_dir = "data/model_data/fusion/baseline_layer1"
best_save_dir = os.path.join(save_dir, "best")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(best_save_dir, exist_ok=True)

origin_data_file = os.path.join(save_dir, "origin_data.txt")
title_predict_file = os.path.join(save_dir, "title_predict.txt")
attr_predict_file = os.path.join(save_dir, "attr_predict.txt")

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

# 生成训练数据
# fine50000 = "data/tmp_data/unequal_processed_data/fine50000.txt"
# rets = []
# with open(fine50000, "r") as f:
#     for line in tqdm(f):
#         data = json.loads(line)
#         # TODO 生成负例
#         rets.append(json.dumps(data, ensure_ascii=False) + "\n")

# # 保存文件
# with open(os.path.join(origin_data_file, "fusion_origin.txt"), "w") as f:
#     f.writelines(rets)

# 执行预测

# 读取预测文件
origin_data = []
title_predict = []
attr_predict = []

for split_index in range(4):
    with open(f"data/tmp_data/fusion/no_feat/pos_{split_index}.txt", "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            origin_data.append(item)

    with open(f"data/tmp_data/fusion/predict/title_{split_index}.txt", "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            title_predict.append(item)

    with open(f"data/tmp_data/fusion/predict/attr_{split_index}.txt", "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            attr_predict.append(item)
origin_data = np.array(origin_data)
title_predict = np.array(title_predict)
attr_predict = np.array(attr_predict)

# for split_index in range(4):
#     split_origin_data = []
#     with open(f"data/tmp_data/fusion/no_feat/pos_{split_index}.txt", "r") as f:
#         for line in tqdm(f):
#             item = json.loads(line)
#             split_origin_data.append(item)
#     origin_data.append(np.array(split_origin_data))

#     split_title_predict = []
#     with open(f"data/tmp_data/fusion/predict/title_{split_index}.txt", "r") as f:
#         for line in tqdm(f):
#             item = json.loads(line)
#             split_title_predict.append(item)
#     title_predict.append(np.array(split_title_predict))

#     split_attr_predict = []
#     with open(f"data/tmp_data/fusion/predict/attr_{split_index}.txt", "r") as f:
#         for line in tqdm(f):
#             item = json.loads(line)
#             split_attr_predict.append(item)
#     attr_predict.append(np.array(split_attr_predict))


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

        new_title_predict = model(origin_predict, mask)

        title_label = label[:, 0]
        loss = loss_fn(new_title_predict, title_label)
        loss_list.append(loss.mean().cpu())

        predict = new_title_predict

        predict[predict > threshold] = 1
        predict[predict <= threshold] = 0

        correct += torch.sum(title_label == predict).cpu()
        total += len(title_label)

    acc = correct / total
    return acc.item(), np.mean(loss_list)


def simple_evaluate(dataloader):
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        origin_predict, label, mask = batch

        origin_title_predict = origin_predict[:, 0]

        origin_title_predict[origin_title_predict > threshold] = 1
        origin_title_predict[origin_title_predict <= threshold] = 0

        title_label = label[:, 0]
        correct += torch.sum(title_label == origin_title_predict).cpu()
        total += len(title_label)

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
# data_index = np.arange(len(origin_data[0]))
data_index = np.arange(len(origin_data))
random.shuffle(data_index)
kf = KFold(n_splits=args.fold, shuffle=False)
for fold_id, (train_index, val_index) in enumerate(kf.split(data_index)):
    if fold_id == args.fold_id:
        # 划分训练集 测试集
        # all_origin_data = get_all_split_by_index(origin_data, data_index)
        # all_title_predict = get_all_split_by_index(title_predict, data_index)
        # all_attr_predict = get_all_split_by_index(attr_predict, data_index)

        # train_origin_data = get_all_split_by_index(origin_data, train_index)
        # train_title_predict = get_all_split_by_index(title_predict, train_index)
        # train_attr_predict = get_all_split_by_index(attr_predict, train_index)

        # val_origin_data = get_all_split_by_index(origin_data, val_index)
        # val_title_predict = get_all_split_by_index(title_predict, val_index)
        # val_attr_predict = get_all_split_by_index(attr_predict, val_index)

        all_origin_data = origin_data
        all_title_predict = title_predict
        all_attr_predict = attr_predict

        train_origin_data = origin_data[train_index]
        train_title_predict = title_predict[train_index]
        train_attr_predict = attr_predict[train_index]

        val_origin_data = origin_data[val_index]
        val_title_predict = title_predict[val_index]
        val_attr_predict = attr_predict[val_index]

        all_dataset = dataset(
            key_attr,
            all_origin_data,
            all_title_predict,
            all_attr_predict,
        )

        train_dataset = dataset(
            key_attr,
            train_origin_data,
            train_title_predict,
            train_attr_predict,
        )

        val_dataset = dataset(
            key_attr,
            val_origin_data,
            val_title_predict,
            val_attr_predict,
        )

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
        model = FusionTitleMlp()
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

                new_title_predict = model(origin_predict, mask)

                title_label = label[:, 0]

                loss = loss_fn(new_title_predict, title_label)

                loss.backward()
                optimizer.step()

                predict = new_title_predict
                predict[predict > threshold] = 1
                predict[predict <= threshold] = 0

                correct += torch.sum(title_label == predict).cpu()
                total += len(title_label)

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
                best_save_path = best_save_dir + f"_acc_fold{args.fold_id}.pth"
                torch.save(model.state_dict(), best_save_path)

            if evl_loss < min_loss:
                min_loss = evl_loss
                best_save_path = best_save_dir + f"_loss_fold{args.fold_id}.pth"
                torch.save(model.state_dict(), best_save_path)

            logging.info(f"max acc: {max_acc} min loss: {min_loss}")