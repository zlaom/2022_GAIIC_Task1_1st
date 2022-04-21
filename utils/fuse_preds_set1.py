import json 
from tqdm import tqdm 
import copy 

# 融合两个单独的文件
title_file = '../pred_title.txt'
attr_file = '../pred_attr.txt'
out_file = '../pred_title_attr_set1.txt'

attrs = []
rets = []
# 先读入attr预测结果
with open(attr_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        attrs.append(data)
# 融合title预测结果
with open(title_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        attrs[i]['match']['图文'] = data['match']['图文']
        if data['match']['图文'] == 1:
            attr_match_copy = copy.deepcopy(attrs[i]['match'])
            for query, pred in attr_match_copy.items():
                if query != '图文':
                    attrs[i]['match'][query] = 1
        rets.append(json.dumps(attrs[i], ensure_ascii=False)+'\n')
        
with open(out_file, 'w') as f:
    f.writelines(rets)