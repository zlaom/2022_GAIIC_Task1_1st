import json 
from tqdm import tqdm 

# 融合两个单独的文件
title_file = '../pred_title.txt'
attr_file = '../pred_attr.txt'
out_file = '../pred_title_attr.txt'

attrs = []
rets = []
with open(attr_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        attrs.append(data)
with open(title_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        attrs[i]['match']['图文'] = data['match']['图文']
        rets.append(json.dumps(attrs[i], ensure_ascii=False)+'\n')
        
with open(out_file, 'w') as f:
    f.writelines(rets)