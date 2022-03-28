import torch
import numpy as np
import tqdm
import json
import argparse
import os
import yaml
from models.gaiic_model import ITM_Model


class GaiicDataset(torch.utils.data.Dataset):
    def __init__(self, data, attr_idx=None) -> None:
        super().__init__()
        self.data = data
        
    
    def __getitem__(self, index):
        dic = {}
        dic['match_label'] = self.data[index]['match']['图文'] # 图文匹配的标签
        dic['title'] = self.data[index]['title']
        dic['feature'] = self.data[index]['feature']
        
        return dic

    def __len__(self):
        return len(self.data)

def test_epoch(model, val_dataloader, loss_fn, device):
    torch.cuda.empty_cache()
    model.eval()
    val_loss_list = []
    total, correct = 0., 0.

    with torch.no_grad():
        for step, all_dic in tqdm.tqdm(enumerate(val_dataloader)):
            image, text, label = all_dic['feature'], all_dic['title'], all_dic['match_label']
            label = torch.from_numpy(np.array(label)).to(device).long()
            image = torch.stack(image, dim=1).to(device).float()
            output, _ = model(image, text)
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
            # print(correct, total)
            loss = loss_fn(output, label)
            val_loss_list.append(loss.item())
            
        
    # print(correct.item() / total)
    return np.mean(val_loss_list), correct.item() / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Compression Reconstruction')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    args = parser.parse_args()
    yaml_path = args.cfg_file
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    test_config = config['TEST']
    
    data_path = './data/train_coarse.txt'
    pos_data_list = []
    neg_data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data['match']['图文'] == 1:
                pos_data_list.append(data)
            else:
                neg_data_list.append(data)
    np.random.shuffle(pos_data_list)
    new_data = neg_data_list + pos_data_list[:len(neg_data_list)]
    print('Test num = %d', len(new_data))
    test_dataset = GaiicDataset(new_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True)
    attr_path = test_config['ATTR_PATH']
    output_path = test_config['OUT_PATH']

    model_path = test_config['CHECKPOINT_PATH']
    model = ITM_Model(config['MODEL'])
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_loss, test_acc = test_epoch(model, test_loader, loss_fn, device)
    print('test loss is : {:.4f}'.format(test_loss))
    print('test acc is : {:.4f}'.format(test_acc))