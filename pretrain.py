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
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from utils.utils import warmup_lr_schedule, step_lr_schedule
from data_pre.dataset import GaiicDataset
from models.gaiic_model import GaiicModel
from tqdm import tqdm

        
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
    model = GaiicModel(model_cfg)        
    model = model.to(device)
    return model


def get_dataloader(dataset_cfg):
    
    data_train_path = dataset_cfg['TRAIN_PATH']
    data_val_path = dataset_cfg['VAL_PATH']
 
    x_train_list = []
    with open(data_train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            x_train_list.append(data)
    
    x_val_list = []
    with open(data_val_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            x_val_list.append(data)

    
    train_dataset = GaiicDataset(x_train_list)
    val_dataset = GaiicDataset(x_val_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=dataset_cfg['TRAIN_BATCH'], shuffle=True,
        num_workers=dataset_cfg['NUM_WORKERS'], drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=dataset_cfg['VAL_BATCH'], shuffle=False,
        num_workers=dataset_cfg['NUM_WORKERS'], drop_last=False, pin_memory=True)

    return train_loader, val_loader, len(x_train_list), len(x_val_list)


def match_correct(output, labels, threshold=0.):
    new_output = output.detach().cpu().numpy()
    new_output = new_output.reshape(-1)
    labels = labels.detach().cpu().numpy()
    new_labels = labels.reshape(-1)

    correct_1 = np.logical_and(new_output > threshold, new_labels==1)
    correct_0 = np.logical_and(new_output <= threshold, new_labels==-1)
    
    return sum(correct_1 + correct_0)


def train_epoch(epoch, model, train_dataloader, train_num, optimizer, criterion, device):
    torch.cuda.empty_cache()
    model.train()
    
    train_loss_list = []
    correct = 0.0
    total = 0.0
    for step, all_dic in enumerate(train_dataloader):
        optimizer.zero_grad()
        if epoch==0:
            warmup_lr_schedule(optimizer, step, optim_cfg['WARMUP_STEPS'], optim_cfg['WARMUP_LR'], optim_cfg['LR'])
        image, text, label = all_dic['feature'], all_dic['title'], all_dic['cos_label']
        
        # text = torch.from_numpy(np.array(text)).to(device)
        label = torch.from_numpy(np.array(label)).to(device).double()
        image = torch.stack(image, dim=1).to(device)
        output = model(text, image)

        loss = criterion(output, label)
        total += label.size(0)
        correct += match_correct(output, label)

        loss.backward()
        optimizer.step()
        
        train_loss_list.append(loss.item() * image.size(0))
        if (step + 1) % dataset_cfg['LOG_STEP'] == 0:
            logging.info('  Epoch: %d/%s, Step: %d/%d, Train_Loss: %f, Train_Acc: %f',
                         epoch + 1, dataset_cfg['EPOCHS'], step + 1, len(train_dataloader),
                         loss.item(), correct / total)

    train_loss = sum(train_loss_list) / train_num
    return train_loss


def test_epoch(model, val_dataloader, val_num, criterion, device):
    torch.cuda.empty_cache()
    model.eval()
    total, correct = 0, 0

    test_loss_list = []
    with torch.no_grad():
        for step, all_dic in tqdm(enumerate(val_dataloader)):
            image, text, label = all_dic['feature'], all_dic['title'], all_dic['cos_label']
            # text = torch.from_numpy(np.array(text)).to(device)
            label = torch.from_numpy(np.array(label)).to(device).double()
            image = torch.stack(image, dim=1).to(device)
            output = model(text, image)

            loss = criterion(output, label)
            total += label.size(0)
            correct += match_correct(output, label)
            
            test_loss_list.append(loss.item() * image.size(0))

    test_loss = sum(test_loss_list) / val_num
    return test_loss, correct / total


def train(model_cfg, dataset_cfg, optim_cfg, device):
    set_seed_logger(dataset_cfg)
    output_folder = os.path.join(dataset_cfg['OUT_PATH'], 'pretrain')
    os.makedirs(output_folder, exist_ok=True)

    model = init_model(model_cfg, device)
    train_dataloader, val_dataloader, train_num, val_num = get_dataloader(dataset_cfg)
    num_train_optimization_steps = len(train_dataloader) * dataset_cfg['EPOCHS']
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=optim_cfg['LR'], weight_decay=optim_cfg['WEIGHT_DECAY'])
    
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=dataset_cfg['OUT_PATH'])
    logging.info('Dataset config = %s', str(dict(dataset_cfg)))
    logging.info('Train num = %d', train_num)
    logging.info('Test num = %d', val_num)
    logging.info('Num steps = %d', num_train_optimization_steps)
    

    for epoch in range(dataset_cfg['EPOCHS']):
        step_lr_schedule(optimizer, epoch, optim_cfg['LR'], optim_cfg['MIN_LR'], optim_cfg['WEIGHT_DECAY'])
        # test_epoch(model, val_dataloader,  val_num, criterion, device)
        train_loss = train_epoch(epoch, model, train_dataloader,  train_num, optimizer, criterion, device)
        val_loss, acc = test_epoch(model, val_dataloader,  val_num, criterion, device)
        
        writer.add_scalar(f'MSE/train', train_loss, epoch)
        writer.add_scalar(f'MSE/val', val_loss, epoch)
        
        logging.info(' Train Epoch %d/%s Finished | Train Loss: %f | Val Loss: %f | Val acc: %f',
                     epoch + 1, dataset_cfg['EPOCHS'], 
                     train_loss, val_loss, acc)
        
        torch.save(model.state_dict(),
                os.path.join(output_folder, 'Pretrain_epoch{:}_acc{:.4f}_loss{:.4f}_.pth'.format(epoch, acc, val_loss)))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Compression Reconstruction')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    args = parser.parse_args()
    yaml_path = args.cfg_file

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']
    dataset_cfg = config['PRETRAIN']
    optim_cfg = config['OPTIM']

    output_folder = dataset_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    os.system('cp {} {}/config.yaml'.format(yaml_path, output_folder))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, dataset_cfg, optim_cfg, device)
