import json 
from tqdm import tqdm 

# 融合两个单独的文件
# file = 'data/tmp_data/results/test_attr_title3.txt'
file = 'data/tmp_data/results/set1.txt'
gt_file = 'data/submission/mytest5000_hide_gt.txt'

preds = []
with open(file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        preds.append(data)

# 图文匹配
correct = 0
total = 0
with open(gt_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        if data['match']['图文'] == preds[i]['match']['图文']:
            correct += 1
        total += 1
print('Image Text Matching Acc:{:.4f}%'.format(correct / total))


# 属性匹配
correct = 0
total = 0
with open(gt_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        gt_match = data['match']
        match = preds[i]['match']
        for query, value in match.items():
            if query != '图文':
                if value == gt_match[query]:
                    correct += 1
                total += 1
print('Attribute Matching Acc:{:.4f}%'.format(correct / total))

