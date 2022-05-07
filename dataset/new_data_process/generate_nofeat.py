import os
from tqdm import tqdm
import json

data_dir = 'data/new_data/equal_split_word'
save_dir = os.path.join(data_dir, 'nofeat')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
for file in os.listdir(data_dir):
    
    if file.split('.')[-1] == 'txt':
        rets = []
        file_path = os.path.join(data_dir, file)
        save_path = os.path.join(save_dir, file)
        with open(file_path, 'r') as f:
            for i, data in enumerate(tqdm(f)):
                data = json.loads(data)
                del data['feature']
                rets.append(json.dumps(data, ensure_ascii=False)+'\n')

    print(len(rets))
    with open(save_path, 'w') as f:
        f.writelines(rets)
