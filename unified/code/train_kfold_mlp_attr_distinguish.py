import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold

from utils.lr_sched import adjust_learning_rate

import argparse

parser = argparse.ArgumentParser("train_attr", add_help=False)
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--fold", default=5, type=int)
parser.add_argument("--fold_id", default=0, type=int)
args = parser.parse_args()

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

batch_size = 512
max_epoch = 200
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
dropout = 0.5

root_dir = f"data/model_data/attr_dis2_simple_mlp_add_{args.fold}fold_e{max_epoch}_b{batch_size}_drop{dropout}/"
save_dir = f"{root_dir}/fold{args.fold_id}/"
best_save_dir = f"{root_dir}best/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(best_save_dir):
    os.makedirs(best_save_dir)

save_name = "attr_model"

# 设置日志路径
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(save_dir, "train.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

# adjust learning rate
LR_SCHED = False
lr = 1e-4
min_lr = 5e-5
warmup_epochs = 5

# data
input_file = [
    "data/tmp_data/equal_processed_data/coarse89588.txt",
    "data/tmp_data/equal_processed_data/fine45000.txt",
]

# 留下5000测试

vocab_file = "data/tmp_data/vocab/vocab.txt"
vocab_dict_file = "data/tmp_data/vocab/vocab_dict.json"
neg_attr_dict_file = "data/tmp_data/equal_processed_data/neg_attr.json"
attr_to_id = "data/tmp_data/equal_processed_data/attr_to_id.json"
macbert_base_file = "data/pretrain_model/macbert_base"

with open(neg_attr_dict_file, "r") as f:
    neg_attr_dict = json.load(f)

with open(attr_to_id, "r") as f:
    attr_to_id = json.load(f)

# 加载数据
all_items = []
for file in input_file:
    with open(file, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            # 图文必须匹配
            if item["match"]["图文"]:
                all_items.append(item)

            # if len(all_items) > 2000:
            #     break
all_items = np.array(all_items)


def generate_data(input_data):
    output_data = []
    for item in tqdm(input_data):
        # 生成所有离散属性
        for attr_key, attr_value in item["key_attr"].items():
            new_item = {}
            similar_sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]
            # un_similar_sample_attr_list = neg_attr_dict[attr_value][
            #     "un_similar_attr"
            # ]
            # sample_attr_list = (
            #     similar_sample_attr_list + un_similar_sample_attr_list
            # )
            for neg_attr_value in similar_sample_attr_list:
                new_item = {}
                new_item["feature"] = item["feature"]
                new_item["attr_ids"] = [
                    attr_to_id[neg_attr_value],
                    attr_to_id[attr_value],
                ]
                new_item["label"] = 1
                output_data.append(new_item)

                new_item = {}
                new_item["feature"] = item["feature"]
                new_item["attr_ids"] = [
                    attr_to_id[attr_value],
                    attr_to_id[neg_attr_value],
                ]
                new_item["label"] = 0
                output_data.append(new_item)
    return output_data


from dataset.keyattrmatch_dataset import (
    AttrIdDistinguishDataset,
    attr_id_distinguish_collate_fn,
)

dataset = AttrIdDistinguishDataset
collate_fn = attr_id_distinguish_collate_fn

# 划分训练集 测试集
kf = KFold(n_splits=args.fold, shuffle=True, random_state=seed)
kf.get_n_splits()
for fold_id, (train_index, test_index) in enumerate(kf.split(all_items)):
    if fold_id == args.fold_id:
        # dataset
        train_data = generate_data(all_items[train_index])
        val_data = generate_data(all_items[test_index])

        train_dataset = dataset(train_data)
        val_dataset = dataset(val_data)

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
        from model.attr_mlp import AttrIdDistinguishMLP

        model = AttrIdDistinguishMLP(dropout=dropout)
        model.cuda()

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # loss
        loss_fn = nn.CrossEntropyLoss()

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
                labels = labels.cuda()
                logits = model(images, attr_ids)
                logits = torch.softmax(logits, dim=-1)

                predict = torch.argmax(logits, dim=-1)

                loss = loss_fn(logits, labels)
                loss_list.append(loss.mean().cpu())

                correct += torch.sum(labels == predict).cpu()
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
                labels = labels.cuda()
                logits = model(images, attr_ids)
                logits = torch.softmax(logits, dim=-1)
                predict = torch.argmax(logits, dim=-1)

                # train acc
                if (i + 1) % 1000 == 0:
                    train_acc = correct / total
                    correct = 0
                    total = 0
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

                correct += torch.sum(labels == predict).cpu()
                total += len(labels)
                i += 1

                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

            evl_acc, evl_loss = evaluate(model, val_dataloader)
            logging.info(f"eval acc: {evl_acc} loss:{evl_loss}")

            if evl_acc > max_acc:
                max_acc = evl_acc
                # save_path = save_dir + save_name + f"_{epoch}_{acc:.4f}.pth"
                best_save_path = (
                    best_save_dir + save_name + f"_acc_fold{args.fold_id}.pth"
                )
                # torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)

            if evl_loss < min_loss:
                min_loss = evl_loss
                # save_path = save_dir + save_name + f"_{epoch}_{acc:.4f}.pth"
                best_save_path = (
                    best_save_dir + save_name + f"_loss_fold{args.fold_id}.pth"
                )
                # torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)

            logging.info(f"max acc: {max_acc} min loss: {min_loss}")
