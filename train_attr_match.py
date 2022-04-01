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
from data_pre.dataset import GaiicAttrDataset
from models.gaiic_model import BLIP_Model, ITM_ATTR_Model
import tqdm

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]
        
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
    model = ITM_ATTR_Model(model_cfg)
    model = model.to(device)
    return model


def prep_optimizer(optimizer_cfg, model, num_train_optimization_steps):
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    offset_no_decay_param_tp = [(n, p) for n, p in no_decay_param_tp if 'offset.' in n]
    not_offfset_no_decay_param_tp = [(n, p) for n, p in no_decay_param_tp if 'offset.' not in n]

    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    offset_decay_param_tp = [(n, p) for n, p in decay_param_tp if 'offset.' in n]
    not_offfset_decay_param_tp = [(n, p) for n, p in decay_param_tp if 'offset.' not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in offset_no_decay_param_tp], 'weight_decay': optimizer_cfg['WEIGHT_DECAY'], 'lr': optimizer_cfg['LEARNING_RATE'] * optimizer_cfg['COEF_LR']},
        {'params': [p for n, p in not_offfset_no_decay_param_tp], 'weight_decay': optimizer_cfg['WEIGHT_DECAY'], 'lr': optimizer_cfg['LEARNING_RATE']},
        {'params': [p for n, p in offset_decay_param_tp], 'weight_decay': 0.0, 'lr': optimizer_cfg['LEARNING_RATE'] * optimizer_cfg['COEF_LR']},
        {'params': [p for n, p in not_offfset_decay_param_tp], 'weight_decay': 0.0, 'lr': optimizer_cfg['LEARNING_RATE']},
    ]
    
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=optimizer_cfg['LEARNING_RATE'])
    scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer, T_max=num_train_optimization_steps,
                                        T_warmup=int(optimizer_cfg['WARMUP_PROPORTION'] * num_train_optimization_steps),
                                        eta_min=optimizer_cfg['ETA_MIN'])

    return optimizer, scheduler

def get_dataloader(dataset_cfg):
    
    data_path_1 = dataset_cfg['DATA_PATH']
    data_list = []
    with open(data_path_1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data_list.append(data)

    l = int(len(data_list) * dataset_cfg['RATIO'])
    np.random.shuffle(data_list)
    x_train_list = data_list[:l]
    x_val_list = data_list[l:]

    train_dataset = GaiicAttrDataset(x_train_list,)
    val_dataset = GaiicAttrDataset(x_val_list,)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=dataset_cfg['TRAIN_BATCH'], shuffle=True,
        num_workers=dataset_cfg['NUM_WORKERS'], drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=dataset_cfg['VAL_BATCH'], shuffle=False,
        num_workers=dataset_cfg['NUM_WORKERS'], drop_last=False, pin_memory=True)

    return train_loader, val_loader, len(x_train_list), len(x_val_list)


def train_epoch(epoch, model, train_dataloader, train_num, optimizer,scheduler, loss_fn, device):
    torch.cuda.empty_cache()
    model.train()
    
    train_loss_list = []
    total, correct = 0., 0.
    for step, all_dic in enumerate(train_dataloader):
        optimizer.zero_grad()
        # if epoch==1000:
        #     warmup_lr_schedule(optimizer, step, optim_cfg['WARMUP_STEPS'], optim_cfg['WARMUP_LR'], optim_cfg['LR'])
        # 

        image, text = all_dic['feature'], all_dic['attr']
        label = torch.from_numpy(np.array(all_dic['attr_match'])).to(device).long()
        image = torch.stack(image, dim=1).to(device).float()
        output = model(image, text)

        _, predicted = torch.max(output.data, 1)
        loss = loss_fn(output, label)
        total += label.size(0)
        correct += (predicted == label).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_list.append(loss.item() * image.size(0))
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
        for step, all_dic in tqdm.tqdm(enumerate(val_dataloader)):
            image, text = all_dic['feature'], all_dic['attr']
            label = torch.from_numpy(np.array(all_dic['attr_match'])).to(device).long()
            image = torch.stack(image, dim=1).to(device).float()
            output = model(image, text)

            _, predicted = torch.max(output.data, 1)
            
            total += label.size(0)
            correct += (predicted == label).sum()
            loss = loss_fn(output, label)
            val_loss_list.append(loss.item())
            
        
    return np.mean(val_loss_list), correct.item() / total


def train(model_cfg, dataset_cfg, optimizer_cfg, device):
    set_seed_logger(dataset_cfg)
    output_folder = os.path.join(dataset_cfg['OUT_PATH'], 'train')
    os.makedirs(output_folder, exist_ok=True)

    model = init_model(model_cfg, device)
    train_dataloader, val_dataloader, train_num, val_num = get_dataloader(dataset_cfg)
    
    num_train_optimization_steps = len(train_dataloader) * dataset_cfg['EPOCHS']
    optimizer, scheduler = prep_optimizer(optimizer_cfg, model, num_train_optimization_steps)
    #optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    # weight_decay=optim_cfg['WEIGHT_DECAY']
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0.)
    loss_fn_1 = nn.CrossEntropyLoss().cuda()
    writer = SummaryWriter(log_dir=dataset_cfg['OUT_PATH'])
    logging.info('Dataset config = %s', str(dict(dataset_cfg)))
    logging.info('Train num = %d', train_num)
    logging.info('Test num = %d', val_num)
    logging.info('Num steps = %d', num_train_optimization_steps)
    

    for epoch in range(dataset_cfg['EPOCHS']):
        #step_lr_schedule(optimizer, epoch, optim_cfg['LR'], optim_cfg['MIN_LR'], optim_cfg['WEIGHT_DECAY'])
        # test_epoch(model, val_dataloader, loss_fn_1, device)
        train_loss = train_epoch(epoch, model, train_dataloader, train_num, optimizer,scheduler, loss_fn_1,  device)
        val_loss, acc = test_epoch(model, val_dataloader, loss_fn_1, device)
        writer.add_scalar(f'CE/train', train_loss, epoch)
        writer.add_scalar(f'ACC/val', acc, epoch)
        
        logging.info(' Train Epoch %d/%s Finished | Train Loss: %f | Val loss: %f | Val acc: %f',
                     epoch + 1, dataset_cfg['EPOCHS'], 
                     train_loss, val_loss,  acc)
        
        torch.save(model.state_dict(),
                os.path.join(output_folder, 'ATTR_Train_epoch{:}_val_loss{:.4f}_val_acc{:.4f}_.pth'.format(epoch, val_loss, acc)))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Compression Reconstruction')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    args = parser.parse_args()
    yaml_path = args.cfg_file

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']['ATTR']
    dataset_cfg = config['ATTR_TRAIN']
    optim_cfg = config['OPTIMIZER']
    
    output_folder = dataset_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    os.system('cp {} {}/config.yaml'.format(yaml_path, output_folder))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, dataset_cfg, optim_cfg, device)
