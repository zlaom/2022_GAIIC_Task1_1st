import os 
from tqdm import tqdm
import json
import yaml

yaml_path = '../new_config.yaml'
with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    save_dir = config['data_processed_3']['save_dir']

# 生成title需要的数据划分
print('divide data for title matching training...')
# [fine40000, fine9000, fine700]
title_dir = os.path.join(save_dir, 'title')
if not os.path.exists(title_dir):
    os.makedirs(title_dir)

fine_file = os.path.join(save_dir, 'fine50000.txt')
save_fine_file_pretrain = os.path.join(title_dir, 'fine40000.txt')
save_fine_file_train = os.path.join(title_dir, 'fine10000.txt')

rets_pretrain = []
rets_train = []


with open(fine_file, 'r') as f:
    for i, line in enumerate(tqdm(f)):
        item = json.loads(line)
        # del item['feature']
        if len(rets_pretrain) < 40000:
            rets_pretrain.append(json.dumps(item, ensure_ascii=False)+'\n')
        elif len(rets_train) < 10000:
            rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
        
        
with open(save_fine_file_pretrain, 'w') as f:
    f.writelines(rets_pretrain)
with open(save_fine_file_train, 'w') as f:
    f.writelines(rets_train)


coarse_file = os.path.join(save_dir, 'coarse10412.txt')
save_coarse_file_train = os.path.join(title_dir, 'coarse10000.txt')

rets_train = []
for file in coarse_file.split(','):
    with open(file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            item = json.loads(line)
            # del item['feature']
            if len(rets_train) < 10000:
                rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
    
        

with open(save_coarse_file_train, 'w') as f:
    f.writelines(rets_train)




