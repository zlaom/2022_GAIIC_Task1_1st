from ntpath import join
import os
import json
import numpy as np 
from tqdm import tqdm 

for file_index in range(4):

    title_file = f'output/fusion/predict/no_pos_title_{file_index}.txt'
    attr_file = f'output/fusion/predict/no_pos_attr_{file_index}.txt'
    
    title_data = []
    attr_data = []
    
    with open(title_file, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            title_data.append(item)
            
    with open(attr_file, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            attr_data.append(item)
        
    
    for index in range(len(title_data)):
        title_item  = title_data[index]
        attr_item  = attr_data[index]
        attr_item["pred"]["图文"] = title_item["pred"]["图文"]
    
    
    all_data = []
    for data in   attr_data:
         all_data.append(json.dumps(data, ensure_ascii=False)+'\n')

    out_dir = 'output/fusion/predict'
    out_file = os.path.join(out_dir, f"no_pos_merge_{file_index}.txt")

    with open(out_file, 'w') as f:
        f.writelines(all_data)


