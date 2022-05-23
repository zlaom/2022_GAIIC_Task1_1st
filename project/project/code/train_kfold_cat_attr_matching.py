import json
import os
import argparse
import random
import torch
import torch.optim as optim
import numpy as np
import time

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils.utils import adjust_learning_rate, my_custom_logger
from attr_config import *
from model.attr_mlp import FinalCatModel
from dataset.unequal_attr_match_dataset import (
    AttrIdMatchDataset,
    attr_id_match_collate_fn,
)

old_time = time.time()
print("开始训练")

# 训练参数
parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--fold_num", default=10, type=int)
parser.add_argument("--fold_ids", nargs="+", type=int)
args = parser.parse_args()

print(f"Trian folds {args.fold_ids} \n")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = 256
max_epoch = 60
eval_num = 3  # 随机验证次数
pos_rate = 0.47  # 属性正例生成比例
dropout = 0.3  # 图像特征dropout率
threshold = 0.5  # 匹配置信度阈值
LR_SCHED = True  # 余弦学习率
lr = 5e-4
min_lr = 5e-6
warmup_epochs = 3

# 固定随机种子
seed = 11
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = True

# 设置存储目录
log_dir = f"{ATTR_MODEL_SAVE_DIR}/log"
best_save_dir = f"{ATTR_MODEL_SAVE_DIR}/best"
save_name = "attr_model"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(best_save_dir):
    os.makedirs(best_save_dir)

# 加载关系字典以及id映射字典
attr_relation_dict_file = f"{PREPROCESS_DATA_DIR}/attr_relation_dict.json"
attr_to_id = f"{PREPROCESS_DATA_DIR}/attr_to_id.json"

with open(attr_relation_dict_file, "r") as f:
    attr_relation_dict = json.load(f)
with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

# 加载数据
coarse89588 = f"{PREPROCESS_DATA_DIR}/coarse89588.txt"
fine50000 = f"{PREPROCESS_DATA_DIR}/fine50000.txt"

all_item_data = []
with open(coarse89588, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        # 训练集图文必须匹配
        if item["match"]["图文"]:
            all_item_data.append(item)

with open(fine50000, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        # 训练集图文必须匹配
        if item["match"]["图文"]:
            all_item_data.append(item)

all_item_data = np.array(all_item_data)

dataset = AttrIdMatchDataset
collate_fn = attr_id_match_collate_fn


# 划分KFold训练集 验证集
kf = KFold(n_splits=args.fold_num, shuffle=True, random_state=seed)

for fold_id, (train_index, test_index) in enumerate(kf.split(all_item_data)):
    if fold_id in args.fold_ids:
        # 设置训练随机种子
        torch.manual_seed(seed + fold_id)
        np.random.seed(seed + fold_id)
        random.seed(seed + fold_id)

        logger = my_custom_logger(os.path.join(log_dir, f"train{fold_id}.log"))

        logger.info(f"Begin train fold {fold_id}")

        # 构建Dataloader
        train_data = all_item_data[train_index]
        val_data = all_item_data[test_index]

        logger.info(f"train_data len {len(train_data)}")
        logger.info(f"val_data len {len(val_data)}")

        train_dataset = dataset(train_data, attr_relation_dict, attr_to_id, pos_rate)
        val_dataset = dataset(val_data, attr_relation_dict, attr_to_id, pos_rate)

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

        # 加载模型
        model = FinalCatModel(attr_num=80, dropout=dropout)
        print("model param num", sum(param.numel() for param in model.parameters()))
        model.cuda()

        # 定义优化器以及Loss
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # 验证函数
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

        # 训练
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

                if (i + 1) % 100 == 0:
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

            evl_acc, evl_loss = [], []
            for _ in range(eval_num):
                _evl_acc, _evl_loss = evaluate(model, val_dataloader)
                evl_acc.append(_evl_acc)
                evl_loss.append(_evl_loss)
            logger.info(f"eval acc: {evl_acc} loss:{evl_loss}")
            evl_acc, evl_loss = np.mean(evl_acc), np.mean(evl_loss)
            logger.info(f"eval mean acc: {evl_acc} mean loss:{evl_loss}")

            # 保存acc最小
            # if evl_acc > max_acc:
            #     max_acc = evl_acc
            #     best_save_path = os.path.join(
            #         best_save_dir, save_name + f"_acc_fold{fold_id}.pth"
            #     )
            #     torch.save(model.state_dict(), best_save_path)

            # 保存loss最小
            if evl_loss < min_loss:
                min_loss = evl_loss
                best_save_path = os.path.join(
                    best_save_dir, save_name + f"_loss_fold{fold_id}.pth"
                )
                torch.save(model.state_dict(), best_save_path)

            logger.info(f"max acc: {max_acc} min loss: {min_loss}")

current_time = time.time()
print(f"训练耗时：{str(current_time - old_time)}s")
