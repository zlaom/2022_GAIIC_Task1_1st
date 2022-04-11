import json
import numpy as np
from tqdm import tqdm
from random import choice

def get_random_key(keys, ratio=0.7):
    list_ratio = [0.3, 0.5]
    ratio = choice(list_ratio)
    # ratio = 0.3
    l = int(len(keys) * ratio)
    if l == 0:
        l = 1
    np.random.shuffle(keys)
    return keys[:l]

def get_title_mask(title, key, val, attr_key):
    # 随机替换title的某个属性值导致图文不匹配
    values = attr_key[key]
    key_index = 0
    for i in range(len(values)):
        if val in values[i]:
            key_index = i
            break
    new_index = np.random.randint(len(values))
    while new_index == key_index:
        new_index = np.random.randint(len(values))
    sub_val = values[new_index]
    new_sub_val = sub_val[np.random.randint(len(sub_val))]

    return title.replace(val, new_sub_val, 1)

with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)


pos_match_list = []
neg_match_list = []
finetune_data = []
def get_all_match(path):
    # 得到图文不匹配做预训练
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        new_dic = {}
        for line in tqdm(lines):
            data = json.loads(line)
            pos_dic = {}
            pos_dic['feature'] = data['feature']
            title = data['title']
            # 去掉数字和 年
            title = ''.join([ch for ch in title if (not ch.isdigit()) and (ch != '年')])
            pos_dic['title'] = title
            pos_dic['all_match'] = 1
            pos_match_list.append(json.dumps(pos_dic, ensure_ascii=False)+'\n')

            new_dic = {}
            new_dic['feature'] = data['feature']
            new_dic['title'] = data['title']
            #print(new_dic['title'])
            keys = get_random_key([x for x in data['match'].keys() if x != '图文'])
            for key in keys:
                new_dic['title'] = get_title_mask(new_dic['title'], key, data['key_attr'][key], attr_key)
            new_dic['all_match'] = 0
            neg_match_list.append(json.dumps(new_dic, ensure_ascii=False)+'\n')

get_all_match('./data/pos_coarse_attr.txt')
get_all_match('./data/train_fine.txt')

print(len(pos_match_list) == len(neg_match_list))
finetune_data = []

# 取出图文不匹配的1w负例, 在加上1w正例做finetune
with open('./data/neg_coarse.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        new_dic = {}
        new_dic['feature'] = data['feature']
        new_dic['title'] = data['title']
        new_dic['all_match'] = 0
        finetune_data.append(json.dumps(new_dic, ensure_ascii=False)+'\n')

l = len(finetune_data)
print(l)
finetune_data = finetune_data + pos_match_list[:l]
train_data = pos_match_list[l:] + neg_match_list

# 打乱图文，属性匹配的数据集
np.random.shuffle(finetune_data)
np.random.shuffle(train_data)
print(len(finetune_data))
print(len(train_data))

# 保存数据
with open('./data/train_all_match_0.3_0.5.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_data)
with open('./data/finetune_all_match_0.3_0.5.txt', 'w', encoding='utf-8') as f:
    f.writelines(finetune_data)
