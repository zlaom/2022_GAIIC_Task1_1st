import json
import random
import numpy as np
import os
from tqdm import tqdm

# 数据生成
fine_file = 'data/new_data/equal_split_word/fine50000.txt'
coarse_file = 'data/new_data/equal_split_word/coarse10412.txt'
save_path = 'data/new_data/divided/title/shuffle'
seed5000 = 2022
seed_list = np.arange(5) + 2022

fine_data = []
coarse_data = []

with open(fine_file, "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        fine_data.append(json.dumps(data, ensure_ascii=False) + "\n")

with open(coarse_file, "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        coarse_data.append(json.dumps(data, ensure_ascii=False) + "\n")

random.seed(seed5000)
random.shuffle(fine_data)
fine5000 = fine_data[45000:]
with open(os.path.join(save_path,f'seed{seed5000}_fine5000.txt'), "w") as f:
    f.writelines(fine5000)
    
fine_data = fine_data[:45000]

# 随机划分5次数据
for seed in seed_list:
    item_save_path = os.path.join(save_path, f"seed_{seed}")
    os.makedirs(item_save_path, exist_ok=True)

    random.seed(int(seed))
    random.shuffle(fine_data)
    random.shuffle(coarse_data)

    fine35300 = fine_data[:35300]
    with open(os.path.join(item_save_path,'fine35300.txt'), "w") as f:
        f.writelines(fine35300)

    fine700 = fine_data[35300:35300+700]
    with open(os.path.join(item_save_path,'fine700.txt'), "w") as f:
        f.writelines(fine700)

    fine9000 = fine_data[35300+700:]
    with open(os.path.join(item_save_path,'fine9000.txt'), "w") as f:
        f.writelines(fine9000)

    coarse9000 = coarse_data[:9000]
    with open(os.path.join(item_save_path,'coarse9000.txt'), "w") as f:
        f.writelines(coarse9000)

    coarse1412 = coarse_data[9000:]
    with open(os.path.join(item_save_path,'coarse1412.txt'), "w") as f:
        f.writelines(coarse1412)