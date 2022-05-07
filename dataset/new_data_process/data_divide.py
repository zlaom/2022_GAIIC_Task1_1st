import os
from tqdm import tqdm 
import json

fine_file = 'data/new_data/equal_split_word/fine50000.txt'
coarse_neg_file = 'data/new_data/equal_split_word/coarse10412.txt'

SAVE_DIR = 'data/new_data/divided'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
# [fine] 划分train val数据
# print('divide fine data...')
# fine_path = os.path.join(SAVE_DIR, 'fine50000.txt')
# fine_train_path = os.path.join(SAVE_DIR, 'fine45000.txt')
# fine_val_path = os.path.join(SAVE_DIR, 'fine5000.txt')

# train_rets = []
# val_rets = []

# with open(fine_path, 'r') as f:
#     for i, data in enumerate(tqdm(f)):
#         data = json.loads(data)
#         if len(train_rets) < 45000:      
#             train_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
#         else:
#             val_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
# print(len(train_rets))
# print(len(val_rets))

# with open(fine_train_path, 'w') as f:
#     f.writelines(train_rets)
# with open(fine_val_path, 'w') as f:
#     f.writelines(val_rets)
    
    
    
# --------------生成title需要的数据划分------------------------------ #
print('divide data for title matching training...')
# [fine40000, fine9000, fine700]
title_dir = os.path.join(SAVE_DIR, 'title')
if not os.path.exists(title_dir):
    os.makedirs(title_dir)

save_fine_file_pretrain = os.path.join(title_dir, 'fine40000.txt')
save_fine_file_train = os.path.join(title_dir, 'fine9000.txt')
save_fine_file_val = os.path.join(title_dir, 'fine700.txt')

rets_pretrain = []
rets_train = []
rets_val = []
for file in fine_file.split(','):
    with open(file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            item = json.loads(line)
            # del item['feature']
            if len(rets_pretrain) < 40000:
                rets_pretrain.append(json.dumps(item, ensure_ascii=False)+'\n')
            elif len(rets_train) < 9000:
                rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
            elif len(rets_val) < 700:
                rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        
with open(save_fine_file_pretrain, 'w') as f:
    f.writelines(rets_pretrain)
with open(save_fine_file_train, 'w') as f:
    f.writelines(rets_train)
with open(save_fine_file_val, 'w') as f:
    f.writelines(rets_val)
    
    
    
# [coarse9000 coarse1412]
print('divide coarse10412...')
save_coarse_file_train = os.path.join(title_dir, 'coarse9000.txt')
save_coarse_file_val = os.path.join(title_dir, 'coarse1412.txt')

rets_train = []
rets_val = []
for file in coarse_neg_file.split(','):
    with open(file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            item = json.loads(line)
            # del item['feature']
            if len(rets_train) < 9000:
                rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
            elif len(rets_val) < 1412:
                rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        

with open(save_coarse_file_train, 'w') as f:
    f.writelines(rets_train)
with open(save_coarse_file_val, 'w') as f:
    f.writelines(rets_val)



# 生成attr需要的数据划分
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