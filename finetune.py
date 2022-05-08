from cProfile import label
import json
import os
import sys
import time
import yaml
import math
import random
import logging
import argparse
import numpy as np
import scipy.io as scio
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from utils.utils import warmup_lr_schedule, step_lr_schedule

from data_pre.cls_match_dataset import ITMDataset, cls_collate_fn, ITMAttrDataset
from models.fuse_model import FuseModel
from models.hero_bert.bert_config import BertConfig
import tqdm

        
def set_seed_logger(dataset_cfg):
    random.seed(dataset_cfg['SEED'])
    os.environ['PYTHONHASHSEED'] = str(dataset_cfg['SEED'])
    np.random.seed(dataset_cfg['SEED'])
    torch.manual_seed(dataset_cfg['SEED'])
    torch.cuda.manual_seed(dataset_cfg['SEED'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    logging.basicConfig(level=logging.INFO,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(dataset_cfg['OUT_PATH'], 'train.log')),
                            logging.StreamHandler(sys.stdout)])


def init_model(model_cfg, device):
    split_config = BertConfig(num_hidden_layers=0)
    fuse_config = BertConfig(num_hidden_layers=8)
    model = FuseModel(split_config, fuse_config, 'data/split_word/vocab/vocab.txt', n_img_expand=6)
    model.load_state_dict(torch.load('checkpoints/train/all_mean_split_43_h8_epd6_best_acc.pth'))
    model = model.to(device)
    return model


def get_dataloader(dataset_cfg):
    
    train_path = './data/equal_split_word/title/43_coarse9000.txt,./data/equal_split_word/title/43_fine9000.txt'
    val_path = './data/equal_split_word/title/43_fine700.txt,./data/equal_split_word/title/43_coarse1412.txt'    

    train_dataset = ITMDataset(train_path, )
    val_dataset = ITMDataset(val_path, )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=dataset_cfg['TRAIN_BATCH'], shuffle=True,
        num_workers=dataset_cfg['NUM_WORKERS'], drop_last=True, pin_memory=True, collate_fn=cls_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=dataset_cfg['VAL_BATCH'], shuffle=False,
        num_workers=dataset_cfg['NUM_WORKERS'], drop_last=False, pin_memory=True, collate_fn=cls_collate_fn)

    return train_loader, val_loader, 18000, 2124


def train_epoch(epoch, model, train_dataloader, train_num, optimizer, loss_fn, device):
    torch.cuda.empty_cache()
    model.train()
    
    train_loss_list = []
    total, correct = 0., 0.
    for step, all_dic in enumerate(train_dataloader):
        optimizer.zero_grad()

        images, splits, labels = all_dic
        images = images.to(device)
        labels = labels.to(device)
        output = model(images, splits)
        
        _, predicted = torch.max(output.data, 1)

        loss = loss_fn(output, labels)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss.backward()
        optimizer.step()
        
        train_loss_list.append(loss.item() * images.size(0))
        if (step + 1) % dataset_cfg['LOG_STEP'] == 0:
            logging.info('  Epoch: %d/%s, Step: %d/%d, Train_Loss: %f, Train_attr_match_Acc: %f ',
                         epoch + 1, dataset_cfg['EPOCHS'], step + 1, len(train_dataloader),
                         loss.item(), correct/total)

    train_loss = sum(train_loss_list) / train_num
    return train_loss


def test_epoch(model, val_dataloader, loss_fn, device):
    torch.cuda.empty_cache()
    model.eval()
    val_loss_list = []
    total, correct = 0., 0.
    with torch.no_grad():
        for all_dic in tqdm.tqdm(val_dataloader):
            images, splits, labels = all_dic
            images = images.to(device)
            labels = labels.to(device)

            output = model(images, splits)
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss = loss_fn(output, labels)
            val_loss_list.append(loss.item())
            
        
    return np.mean(val_loss_list), correct.item() / total


def train(model_cfg, dataset_cfg, optim_cfg, device):
    set_seed_logger(dataset_cfg)
    output_folder = os.path.join(dataset_cfg['OUT_PATH'], 'finetune')
    os.makedirs(output_folder, exist_ok=True)

    model = init_model(model_cfg, device)
    train_dataloader, val_dataloader, train_num, val_num = get_dataloader(dataset_cfg)
    num_train_optimization_steps = len(train_dataloader) * dataset_cfg['EPOCHS']
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)

    loss_fn_1 = nn.CrossEntropyLoss().cuda()
    writer = SummaryWriter(log_dir=dataset_cfg['OUT_PATH'])
    logging.info('Dataset config = %s', str(dict(dataset_cfg)))
    logging.info('Train num = %d', train_num)
    logging.info('Test num = %d', val_num)
    logging.info('Num steps = %d', num_train_optimization_steps)
    

    for epoch in range(dataset_cfg['EPOCHS']):
        # step_lr_schedule(optimizer, epoch, optim_cfg['LR'], optim_cfg['MIN_LR'], optim_cfg['WEIGHT_DECAY'])
        # test_epoch(model, val_dataloader, loss_fn_1, device)
        train_loss = train_epoch(epoch, model, train_dataloader, train_num, optimizer, loss_fn_1,  device)
        val_loss, acc = test_epoch(model, val_dataloader, loss_fn_1, device)
        writer.add_scalar(f'CE/train', train_loss, epoch)
        writer.add_scalar(f'ACC/val', acc, epoch)
        
        logging.info(' Train Epoch %d/%s Finished | Train Loss: %f | Val loss: %f | Val acc: %f',
                     epoch + 1, dataset_cfg['EPOCHS'], 
                     train_loss, val_loss,  acc)
        
        torch.save(model.state_dict(),
                os.path.join(output_folder, 'seed_2_mean_h8_epd6_finetune_best_acc_epoch{:}_val_loss{:.4f}_val_acc{:.4f}_.pth'.format(epoch, val_loss, acc)))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Compression Reconstruction')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    args = parser.parse_args()
    yaml_path = args.cfg_file

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']['ALL_MATCH']
    dataset_cfg = config['FINETUNE']
    optim_cfg = config['OPTIM']
    print(dataset_cfg)
    print('sb ayre')
    output_folder = dataset_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    os.system('cp {} {}/config.yaml'.format(yaml_path, output_folder))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, dataset_cfg, optim_cfg, device)