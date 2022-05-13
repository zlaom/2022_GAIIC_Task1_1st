import os 
from tqdm import tqdm
import json
import yaml
import numpy as np

seed_num = 2022
np.random.seed(seed_num)

#seed_num = 'order'
yaml_path = '../new_config_1.yaml'
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
save_fine_file_pretrain = os.path.join(title_dir, f'{seed_num}_fine40000.txt')
save_fine_file_train = os.path.join(title_dir, f'{seed_num}_fine9000.txt')
save_fine_file_val = os.path.join(title_dir, f'{seed_num}_fine700.txt')

fine_data = []
rets_pretrain = []
rets_train = []
rets_val = []

with open(fine_file, 'r') as f:
    for i, line in enumerate(tqdm(f)):
        item = json.loads(line)
        fine_data.append(json.dumps(item, ensure_ascii=False)+'\n')
        # del item['feature']

np.random.shuffle(fine_data)
rets_pretrain = fine_data[:40000]
rets_train = fine_data[40000:49000]
rets_val = fine_data[49000:49700]
print(len(rets_pretrain))
print(len(rets_train))
print(len(rets_val))        
with open(save_fine_file_pretrain, 'w') as f:
    f.writelines(rets_pretrain)
with open(save_fine_file_train, 'w') as f:
    f.writelines(rets_train)
with open(save_fine_file_val, 'w') as f:
    f.writelines(rets_val)


# [coarse89588 coarse9000 coarse1412]
coarse_file = os.path.join(save_dir, 'coarse10412.txt')
save_coarse_file_train = os.path.join(title_dir, f'{seed_num}_coarse9000.txt')
save_coarse_file_val = os.path.join(title_dir, f'{seed_num}_coarse1412.txt')

coarse_data = []
rets_train = []
rets_val = []
for file in coarse_file.split(','):
    with open(file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            item = json.loads(line)
            # del item['feature']
            coarse_data.append(json.dumps(item, ensure_ascii=False)+'\n')
np.random.shuffle(coarse_data)
rets_train = coarse_data[:9000]
rets_val = coarse_data[9000:]        
print(len(rets_train))
print(len(rets_val))

with open(save_coarse_file_train, 'w') as f:
    f.writelines(rets_train)
with open(save_coarse_file_val, 'w') as f:
    f.writelines(rets_val)



# # 生成attr需要的数据划分
# print('divide data for attr matching training...')
# # [coarse89588 coarse85000 coarse4588]
# attr_dir = os.path.join(save_dir, 'attr')
# if not os.path.exists(attr_dir):
#     os.makedirs(attr_dir)

# coarse_file = os.path.join(save_dir, 'coarse89588.txt')
# save_coarse_file_train = os.path.join(attr_dir, 'coarse85000.txt')
# save_coarse_file_val = os.path.join(attr_dir, 'coarse4588.txt')

# rets_train = []
# rets_val = []
# for file in coarse_file.split(','):
#     with open(file, 'r') as f:
#         for i, line in enumerate(tqdm(f)):
#             item = json.loads(line)
#             # del item['feature']
#             if len(rets_train) < 85000:
#                 rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
#             elif len(rets_val) < 4588:
#                 rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        

# with open(save_coarse_file_train, 'w') as f:
#     f.writelines(rets_train)
# with open(save_coarse_file_val, 'w') as f:
#     f.writelines(rets_val)