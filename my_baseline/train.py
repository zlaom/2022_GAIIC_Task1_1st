import os
import logging
import time
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from dataset import JsonMultiTagsDataset
from torch.utils.data import DataLoader
from model import MutilTagModel
from params import parse_args

# from logger import setup_primary_logging, setup_worker_logging


def train_one_epoch(model, dataloader, optimizer, critic, epoch, args, scheduler=None):
    model.train()
    num_batches_per_epoch = len(dataloader)
    start_time = time.time()
    acc_predict = 0
    all_attr = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        # scheduler(step)
        optimizer.zero_grad()
        images, labels = batch
        images = images.cuda(args.gpu, non_blocking=True).float()
        labels = labels.cuda(args.gpu, non_blocking=True).float()
        predicts = model(images)
        loss = critic(predicts, labels)
        loss.backward()
        optimizer.step()
        predict_tags = predicts > 0.5
        acc_predict += torch.sum(predict_tags[labels == 1] == 1)
        all_attr += torch.sum(labels)

        if (i % 100) == 0:
            acc = acc_predict / all_attr
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{i}/{num_batches_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f}\tAcc: {acc:.6f}"
            )

    end_time = time.time()
    acc = acc_predict / all_attr
    logging.info("batch_time: {}".format(end_time - start_time))
    return acc


def evaluate(model, dataloader, args):
    model.eval()
    acc_predict = 0
    all_attr = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.cuda(args.gpu, non_blocking=True).float()
            labels = labels.cuda(args.gpu, non_blocking=True).float()
            predicts = model(images)
            predict_tags = predicts > 0.5
            acc_predict += torch.sum(predict_tags[labels == 1] == 1)
            all_attr += torch.sum(labels)
    acc = acc_predict / all_attr
    return acc


def main():
    args = parse_args()
    args.log_path = os.path.join(args.logs, args.name, "out.log")
    # if os.path.exists(args.log_path):
    #     print(
    #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
    #     )
    #     return -1
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    args.log_level = logging.DEBUG if args.debug else logging.INFO  # INFO
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(args.log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Params:")
    params_file = os.path.join(args.logs, args.name, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    train_dataset = JsonMultiTagsDataset(args.train_data, args.attr_vals, is_train=True)
    val_dataset = JsonMultiTagsDataset(args.val_data, args.attr_vals, is_train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=128, num_workers=4, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, shuffle=False)

    model = MutilTagModel()
    model.cuda(args.gpu)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    critic = nn.BCELoss()
    best_acc = 0
    for epoch in range(args.epochs):
        train_acc = train_one_epoch(model, train_loader, optimizer, critic, epoch, args)
        val_acc = evaluate(model, val_loader, args)
        logging.info("train_acc{}, val_acc{}".format(train_acc, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            # save model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
            )
        # save latest model
        torch.save(
            {
                "epoch": epoch + 1,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
        )


if __name__ == "__main__":
    main()
