import os
from tqdm import tqdm 
import json
import random 

import argparse 
parser = argparse.ArgumentParser('', add_help=False)
parser.add_argument('--seed', type=int)
args = parser.parse_args()   

seed = args.seed

fine_file = 'temp/tmp_data/lhq_data/equal_split_word/fine50000.txt'
coarse_neg_file = 'temp/tmp_data/lhq_data/equal_split_word/coarse10412.txt'

SAVE_DIR = 'temp/tmp_data/lhq_data/divided/title/seed'+str(seed)+'/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
all_data = []
with open(fine_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        all_data.append(data)
        
# [fine] 划分train val数据
print('divide fine data...')
path1 = os.path.join(SAVE_DIR, 'fine40000.txt')
path3 = os.path.join(SAVE_DIR, 'fine9000.txt')
path4 = os.path.join(SAVE_DIR, 'fine700.txt')

rets1 = []
rets3 = []
rets4 = []

all_data = []
with open(fine_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        all_data.append(data)
        
random.seed(seed)
random.shuffle(all_data)

for data in tqdm(all_data):
    if len(rets1) < 40000:
        rets1.append(json.dumps(data, ensure_ascii=False)+'\n')
    elif len(rets3) < 9000:
        rets3.append(json.dumps(data, ensure_ascii=False)+'\n')
    elif len(rets4) < 700:
        rets4.append(json.dumps(data, ensure_ascii=False)+'\n')

        
print(len(rets1))
print(len(rets3))
print(len(rets4))

with open(path1, 'w') as f:
    f.writelines(rets1)
with open(path3, 'w') as f:
    f.writelines(rets3)
with open(path4, 'w') as f:
    f.writelines(rets4)
    
    
    
print('divide coarse data...')

path1 = os.path.join(SAVE_DIR, 'coarse9000.txt')
path2 = os.path.join(SAVE_DIR, 'coarse1412.txt')

rets1 = []
rets2 = []

all_data = []
with open(coarse_neg_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        all_data.append(data)
        
random.seed(seed)
random.shuffle(all_data)

for data in tqdm(all_data):
    if len(rets1) < 9000:
        rets1.append(json.dumps(data, ensure_ascii=False)+'\n')
    elif len(rets2) < 1412:
        rets2.append(json.dumps(data, ensure_ascii=False)+'\n')

        
print(len(rets1))
print(len(rets2))

with open(path1, 'w') as f:
    f.writelines(rets1)
with open(path2, 'w') as f:
    f.writelines(rets2)