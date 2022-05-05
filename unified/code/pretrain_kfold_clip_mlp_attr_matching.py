import json
import os
import random
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

batch_size = 2048
max_epoch = 2000
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

root_dir = f"data/model_data/pretrain_clip2_attr_simple_mlp_{args.fold}fold_e{max_epoch}_b{batch_size}_drop0/"
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
lr = 1e-5
min_lr = 5e-5
warmup_epochs = 5

# data
input_file = [
    "data/tmp_data/equal_processed_data/coarse89588.txt",
    "data/tmp_data/equal_processed_data/fine50000.txt",
]

vocab_file = "data/tmp_data/vocab/vocab.txt"
vocab_dict_file = "data/tmp_data/vocab/vocab_dict.json"
neg_attr_dict_file = "data/tmp_data/equal_processed_data/neg_attr.json"
attr_to_id = "data/tmp_data/equal_processed_data/attr_to_id.json"
macbert_base_file = "data/pretrain_model/macbert_base"

with open(neg_attr_dict_file, "r") as f:
    neg_attr_dict = json.load(f)

# 加载数据
all_items = []
for file in input_file:
    with open(file, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            # 训练集图文必须匹配
            if item["match"]["图文"]:
                # 生成所有离散属性
                for attr_key, attr_value in item["key_attr"].items():
                    new_item = {}
                    new_item["feature"] = item["feature"]
                    new_item["key"] = attr_key
                    new_item["attr"] = attr_value
                    new_item["label"] = 1
                    all_items.append(new_item)
            # if len(all_items) > 2000:
            #     break
all_items = np.array(all_items)


from dataset.keyattrmatch_dataset import AttrIdMatchDataset2, attr_id_match_collate_fn

dataset = AttrIdMatchDataset2
collate_fn = attr_id_match_collate_fn
# 划分训练集 测试集
kf = KFold(n_splits=args.fold, shuffle=True, random_state=seed)
kf.get_n_splits()
for fold_id, (train_index, val_index) in enumerate(kf.split(all_items)):
    if fold_id == args.fold_id:
        logging.info(f"train attr_num:{len(train_index)}")
        logging.info(f"val attr_num:{len(val_index)}")
        # dataset
        train_data = all_items[train_index]
        val_data = all_items[val_index]

        train_dataset = dataset(train_data, attr_to_id)
        val_dataset = dataset(val_data, attr_to_id)

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
        from model.attr_mlp import CLIP_ATTR_ID_MLP

        model = CLIP_ATTR_ID_MLP()
        model.cuda()

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # loss
        loss_img = nn.CrossEntropyLoss()
        loss_attr = nn.CrossEntropyLoss()

        # evaluate
        @torch.no_grad()
        def evaluate(model, val_dataloader):
            # 重置random种子
            random.seed(2022)
            model.eval()
            cumulative_loss = 0.0
            num_elements = 0.0
            for batch in tqdm(val_dataloader):
                images, attr_ids, labels, _ = batch
                images = images.cuda()
                attr_ids = attr_ids.cuda()
                image_features, attr_features, logit_scale = model(images, attr_ids)

                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ attr_features.t()
                logits_per_attr = logit_scale * attr_features @ image_features.t()
                ground_truth = torch.arange(len(logits_per_image)).long()
                ground_truth = ground_truth.cuda(non_blocking=True)

                total_loss = (
                    loss_img(logits_per_image, ground_truth)
                    + loss_attr(logits_per_attr, ground_truth)
                ) / 2

                batch_size = len(images)
                cumulative_loss += total_loss * batch_size
                num_elements += batch_size

            return cumulative_loss / num_elements

        max_eval_loss = np.inf
        for epoch in range(max_epoch):
            model.train()
            total_loss_list = []

            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if LR_SCHED:
                    lr_now = adjust_learning_rate(
                        optimizer, max_epoch, epoch + 1, warmup_epochs, lr, min_lr
                    )

                images, attr_ids, labels, _ = batch
                images = images.cuda()
                attr_ids = attr_ids.cuda()
                labels = labels.float().cuda()

                image_features, attr_features, logit_scale = model(images, attr_ids)

                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ attr_features.t()
                logits_per_attr = logit_scale * attr_features @ image_features.t()
                ground_truth = torch.arange(len(logits_per_image)).long()
                ground_truth = ground_truth.cuda(non_blocking=True)

                total_loss = (
                    loss_img(logits_per_image, ground_truth)
                    + loss_attr(logits_per_attr, ground_truth)
                ) / 2

                total_loss_list.append(total_loss.mean().item())

                total_loss.backward()
                optimizer.step()

                # train acc
                if (i + 1) % 50 == 0:
                    mean_loss = np.mean(total_loss_list)
                    logging.info(
                        "Epoch:[{}|{}], Training Loss:{:.2f}".format(
                            epoch, max_epoch, mean_loss
                        )
                    )

            eval_loss = evaluate(model, val_dataloader)
            logging.info(f"Eval Loss: {eval_loss}")

            if eval_loss < max_eval_loss:
                max_eval_loss = eval_loss
                # save_path = save_dir + save_name + f"_{epoch}_{eval_loss:.4f}.pth"
                best_save_path = best_save_dir + save_name + f"_fold{args.fold_id}.pth"
                # torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)

            logging.info(f"Best Loss: {max_eval_loss}")
