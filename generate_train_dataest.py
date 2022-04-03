import re
import json
import torch
import numpy as np
from tqdm import tqdm 
import itertools


new_dic = {}
# 生成 attr_values值
with open('./data/attr_to_attrvals.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)
    for key, value in attr_key.items():
        tmp = []
        for v in value:
            if '=' in v:
                tmp.append(v.split('='))
            else:
                tmp.append([v])
        new_dic[key] = tmp


rets = [json.dumps(new_dic, ensure_ascii=False)+'\n']
with open('./data/attr_match.json', 'w', encoding='utf-8') as f:
    f.writelines(rets)

def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r', encoding='utf-8') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict

def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    # return "{}{}".format(attr, ''.join(ret))
    return "{}".format(''.join(ret))  

# load attribute dict
attr_dict_file = "./data/attr_to_attrvals.json"
attr_dict = load_attr_dict(attr_dict_file)

# remove years and get attributes [coarse]
# 裤门襟和闭合方式的属性有重合，所以需要额外的判断机制
coarse_data = './data/train_coarse.txt'
pos_coarse_data = './data/pos_coarse_attr.txt'
neg_coarse_data = './data/neg_coarse.txt'
neg_rets = []
pos_rets = []



querys = attr_dict.keys()
# 得到coarse.txt的属性
with open(coarse_data, 'r', encoding='utf-8') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        title = data['title']
        # 去除title中年份和数字
        title = ''.join([ch for ch in title if (not ch.isdigit()) and (ch != '年')])
        data['title'] = title
        
        if data['match']['图文'] == 1:
            for query in querys:
                values = attr_dict[query]
                if (query == '裤门襟') and ('裤' not in title):
                    continue
                if (query == '闭合方式') and ('裤' in title):
                    continue
                for value in values:
                    if query == '衣长' and '中长款' in title:
                        data['key_attr'][query] = '中长款'
                        data['match'][query] = 1
                        break

                    if query == '裙长' and '中长裙' in title:
                        data['key_attr'][query] = '中长裙'
                        data['match'][query] = 1
                        break

                    if value in title:
                        data['key_attr'][query] =  value
                        data['match'][query] = 1
            if data['key_attr']:    
                pos_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        else:
            neg_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
        
print(len(pos_rets))
print(len(neg_rets))        
with open(pos_coarse_data, 'w', encoding='utf-8') as f:
    f.writelines(pos_rets)
with open(neg_coarse_data, 'w', encoding='utf-8') as f:
    f.writelines(neg_rets)



data_list = []
save_path = './data/all_attr_match.txt'

with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)

def get_dismatch_value(key, val, attr_key):
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

    return new_sub_val


data_list = []
def get_all_attr(path):
    # 得到所有的属性，匹配属性和随机生成不匹配的数据
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            for key, value in data['key_attr'].items():
                new_dic = {}
                new_dic['feature'] = data['feature']
                new_dic['attr'] = key + value
                new_dic['attr_match'] = 1
                data_list.append(json.dumps(new_dic, ensure_ascii=False)+'\n')
                new_dic = {}
                new_dic['feature'] = data['feature']

                new_dic['attr'] = key + get_dismatch_value(key, value, attr_key)
                new_dic['attr_match'] = 0
                data_list.append(json.dumps(new_dic, ensure_ascii=False)+'\n')
    

attr_path_1 = './data/train_fine.txt'
attr_path_2 = './data/pos_coarse_attr.txt'
get_all_attr(attr_path_1)
get_all_attr(attr_path_2)

print(len(data_list))

with open(save_path, 'w', encoding='utf-8') as f:
    f.writelines(data_list)


from random import choice
import json
import numpy as np
from tqdm import tqdm

def get_random_key(keys, ratio=0.7):
    list_ratio = [0.3, 0.5, 0.7, 0.9, 1]
    ratio = choice(list_ratio)
    l = int(len(keys) * ratio)
    if l == 0:
        l = 1
    np.random.shuffle(keys)
    return keys[:l]

with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)

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

pos_match_list = []
neg_match_list = []

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
with open('./data/train_all_match.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_data)
with open('./data/finetune_all_match.txt', 'w', encoding='utf-8') as f:
    f.writelines(finetune_data)


# 属性词典
with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)
new_dic = {}
i = 0
for key, values in attr_key.items():
    for single_lis in values:
        for val in single_lis:
            new_dic[key+val] = i
            i += 1
print(len(new_dic.keys()))

with open('./data/attr_dic.json', 'w', encoding='utf-8') as f:
    json.dump(new_dic, f, ensure_ascii=False)