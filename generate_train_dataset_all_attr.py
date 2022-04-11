import re
import json
import torch
import numpy as np
from tqdm import tqdm 
import itertools
from collections import defaultdict

new_dic = {}
# 生成 attr_values值

# { 裤长:{"短裤": 短裤} }
new_dic_all = {}
with open('./data/attr_to_attrvals.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)
    for key, value in attr_key.items():
        tmp = []
        single_all_dic = {}
        for v in value:
            if '=' in v:
                single_all_dic[v] = v.split('=')[0]
                tmp.append(v.split('='))
            else:
                single_all_dic[v] = v
                tmp.append([v])

            if key == '裙长' and '中长裙' in v:
                single_all_dic[v] = '中裙'
                continue
            if key == '衣长' and '超长款' in v:
                single_all_dic[v] = '超长款'
                continue
        new_dic_all[key] = single_all_dic
        new_dic[key] = tmp


rets = [json.dumps(new_dic, ensure_ascii=False)+'\n']
with open('./data/attr_match.json', 'w', encoding='utf-8') as f:
    f.writelines(rets)

with open('./data/attr_dic_all_1.json', 'w', encoding='utf-8') as f:
    json.dump(new_dic_all, f, ensure_ascii=False)

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
pos_coarse_data = './data/pos_coarse_attr_54.txt'
neg_coarse_data = './data/neg_coarse_54.txt'
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
save_path = './data/all_attr_match_54.txt'

with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_key = json.load(f)

def get_dismatch_value(key, val, attr_key):
    # 负样本属性产生
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
                for all_keys, sing_v in new_dic_all[key].items():
                    if key == '裙长' and value == '中长裙':
                        new_value = '中裙'
                        break
                    elif key == '衣长' and value == '中长款':
                        new_value = '中长款'
                        break
                    else:
                        if value in all_keys:
                            new_value = sing_v
                            break
            
                new_dic['attr'] = key + new_value
                new_dic['attr_match'] = 1
                data_list.append(json.dumps(new_dic, ensure_ascii=False)+'\n')
                new_dic = {}
                new_dic['feature'] = data['feature']
                
                neg_value = get_dismatch_value(key, value, attr_key)

                new_value = ''
                for all_keys, sing_v in new_dic_all[key].items():
                    if key == '裙长' and neg_value == '中长裙':
                        new_value = '中裙'
                        break
                    elif key == '衣长' and neg_value == '中长款':
                        new_value = '中长款'
                        break
                    else:
                        if neg_value in all_keys:
                            new_value = sing_v
                            break

                new_dic['attr'] = key + new_value
                new_dic['attr_match'] = 0
                data_list.append(json.dumps(new_dic, ensure_ascii=False)+'\n')
    

attr_path_1 = './data/train_fine.txt'
attr_path_2 = './data/pos_coarse_attr_54.txt'
get_all_attr(attr_path_1)
get_all_attr(attr_path_2)

print(len(data_list))

with open(save_path, 'w', encoding='utf-8') as f:
    f.writelines(data_list)


new_dic = {}
i = 0
for key, values in new_dic_all.items():
    for all_value, val in values.items():
        new_dic[key+val] = i
        i += 1
print(len(new_dic.keys()))

with open('./data/attr_dic_54.json', 'w', encoding='utf-8') as f:
    json.dump(new_dic, f, ensure_ascii=False)