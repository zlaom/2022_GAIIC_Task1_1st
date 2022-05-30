from cProfile import label
import json
import os
import random
import sys
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold

from utils.lr_sched import adjust_learning_rate

# fuse model
from model.attr_mlp import (
    CatModel,
    DoubleCatModel,
    DoubleCatModel2,
    AttenModel,
    SeAttrIdMatch1,
    SeAttrIdMatch2,
    RestNetAttrIdMatch,
)

from dataset.unequal_keyattr_match_dataset import (
    AllNegAttrIdMatchDataset,
    all_neg_attr_id_match_collate_fn,
)

import argparse

parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--fold", default=10, type=int)
parser.add_argument("--fold_ids", default=[0], nargs="+", type=int)
args = parser.parse_args()

print(f"{args.fold_ids} \n")

# fix the seed for reproducibility
seed = 1212
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

batch_size = 512
max_epoch = 20
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

dropout = 0.3
threshold = 0.5

# root_dir = f"data/model_data/unequal_all_neg_attr/final_cat_origin_mlp_{args.fold}fold_e{max_epoch}_b{batch_size}_drop{dropout}/"
root_dir = f"data/model_data/unequal_all_neg_attr/final_se_origin_mlp_{args.fold}fold_e{max_epoch}_b{batch_size}_drop{dropout}/"
# adjust learning rate
# LR_SCHED = True
# lr = 5e-4
# min_lr = 5e-6
# warmup_epochs = 3

LR_SCHED = True
lr = 5e-4
min_lr = 1e-6
warmup_epochs = 3

# model = SeAttrIdMatch2(attr_num=80, dropout=dropout)
# print("param num", sum(param.numel() for param in model.parameters()))
# exit()

# data

coarse89588 = "data/tmp_data/unequal_processed_data/coarse89588.txt"
fine50000 = "data/tmp_data/unequal_processed_data/fine50000.txt"


attr_relation_dict_file = "data/tmp_data/unequal_processed_data/attr_relation_dict.json"
attr_to_id = "data/tmp_data/unequal_processed_data/attr_to_id.json"

with open(attr_relation_dict_file, "r") as f:
    attr_relation_dict = json.load(f)
with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

# 加载数据
all_item_data = []
with open(coarse89588, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        # 训练集图文必须匹配
        if item["match"]["图文"]:
            all_item_data.append(item)
        # if len(all_item_data) > 2000:
        #     break

with open(fine50000, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        # 训练集图文必须匹配
        if item["match"]["图文"]:
            all_item_data.append(item)
        # if len(all_item_data) > 2000:
        #     break

all_item_data = np.array(all_item_data)

dataset = AllNegAttrIdMatchDataset
collate_fn = all_neg_attr_id_match_collate_fn

# 划分训练集 测试集
kf = KFold(n_splits=args.fold, shuffle=True, random_state=seed)

# 日志函数
def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(asctime)s - %(levelname)s: %(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode="a")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


for fold_id, (train_index, test_index) in enumerate(kf.split(all_item_data)):
    if fold_id in args.fold_ids:
        # 设置路径
        save_dir = f"{root_dir}fold{fold_id}/"
        best_save_dir = f"{root_dir}best/"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(best_save_dir):
            os.makedirs(best_save_dir)

        save_name = "attr_model"

        logger = my_custom_logger(os.path.join(save_dir, f"train{fold_id}.log"))

        logger.info(f"Begin train fold {fold_id}")

        # dataset
        train_data = all_item_data[train_index]
        val_data = all_item_data[test_index]

        logger.info(f"train_data len {len(train_data)}")
        logger.info(f"val_data len {len(val_data)}")

        train_dataset = dataset(train_data, attr_relation_dict, attr_to_id)
        val_dataset = dataset(val_data, attr_relation_dict, attr_to_id)

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

        # model = CatModel(attr_num=80, dropout=dropout)
        model = SeAttrIdMatch2(attr_num=80, dropout=dropout)
        print("param num", sum(param.numel() for param in model.parameters()))
        # exit()
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
            loss_list = []
            for batch in tqdm(val_dataloader):
                images, attr_ids, labels = batch
                images = images.cuda()
                attr_ids = attr_ids.cuda()
                labels = labels.float().cuda()
                logits = model(images, attr_ids)

                predicts = torch.sigmoid(logits)

                predicts[predicts > threshold] = 1
                predicts[predicts <= threshold] = 0

                loss = loss_fn(logits, labels)
                loss_list.append(loss.mean().cpu())

                correct += torch.sum(labels == predicts).cpu()
                total += len(labels)

            acc = correct / total
            return acc.item(), np.mean(loss_list)

        max_acc = 0
        min_loss = np.inf
        last_path = None
        correct = 0
        total = 0
        for epoch in range(max_epoch):
            model.train()

            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if LR_SCHED:
                    lr_now = adjust_learning_rate(
                        optimizer, max_epoch, epoch + 1, warmup_epochs, lr, min_lr
                    )

                images, attr_ids, labels = batch
                images = images.cuda()
                attr_ids = attr_ids.cuda()
                labels = labels.float().cuda()
                logits = model(images, attr_ids)

                # train acc
                if (i + 1) % 500 == 0:
                    train_acc = correct / total
                    correct = 0
                    total = 0
                    if LR_SCHED:
                        logger.info(
                            "Epoch:[{}|{}], Acc:{:.2f}%, LR:{:.2e}".format(
                                epoch, max_epoch, train_acc * 100, lr_now
                            )
                        )
                    else:
                        logger.info(
                            "Epoch:[{}|{}], Acc:{:.2f}%".format(
                                epoch, max_epoch, train_acc * 100
                            )
                        )

                predicts = torch.sigmoid(logits)
                predicts[predicts > threshold] = 1
                predicts[predicts <= threshold] = 0

                correct += torch.sum(labels == predicts).cpu()
                total += len(labels)
                i += 1

                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

            evl_acc, evl_loss = evaluate(model, val_dataloader)
            logger.info(f"eval acc: {evl_acc} loss:{evl_loss}")

            if evl_acc > max_acc:
                max_acc = evl_acc
                # save_path = save_dir + save_name + f"_{epoch}_{acc:.4f}.pth"
                best_save_path = best_save_dir + save_name + f"_acc_fold{fold_id}.pth"
                # torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)

            if evl_loss < min_loss:
                min_loss = evl_loss
                # save_path = save_dir + save_name + f"_{epoch}_{acc:.4f}.pth"
                best_save_path = best_save_dir + save_name + f"_loss_fold{fold_id}.pth"
                # torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)

            logger.info(f"max acc: {max_acc} min loss: {min_loss}")