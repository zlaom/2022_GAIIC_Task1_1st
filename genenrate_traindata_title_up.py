import re
import json
import torch
import numpy as np
from tqdm import tqdm 
import itertools


new_dic = {}
# 生成 attr_values值 { key:[[val_1=,], [val_2=, ]]}
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


dic_rets = [json.dumps(new_dic, ensure_ascii=False)+'\n']
with open('./data/attr_match.json', 'w', encoding='utf-8') as f:
    f.writelines(dic_rets)



def process_title(path):
    # 处理title中引起歧义的两个词和过滤title
    rets = []
    with open(path, 'r') as f:
        for i, data in enumerate(tqdm(f)):
            data = json.loads(data)
            title = data['title']
            # 去除title中年份和数字
            title = ''.join([ch for ch in title if (not ch.isdigit()) and (ch != '年')])

            key_attr = data['key_attr']
            
            # 属性替换
            for query, attr in key_attr.items():
                # 去掉两个特殊的属性
                if query=='衣长' and attr=='中长款':
                    key_attr[query] = '中款'
                    title = title.replace(attr, '中款')
                if query=='裙长' and attr=='中长裙':
                    key_attr[query] = '中裙'
                    title = title.replace(attr, '中裙')
            # 一个高频词的特殊处理
            if '厚度常规' in title:
                title = title.replace('厚度常规', '常规厚度')
            data['key_attr'] = key_attr
            data['title'] = title
            rets.append(json.dumps(data, ensure_ascii=False)+'\n')
    return rets
          

def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r', encoding='utf-8') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict


def generate_key_attr(path):
    attr_dict = load_attr_dict("./data/attr_to_attrvals.json")
    querys = attr_dict.keys()
    # 得到coarse.txt的属性
    pos_attr_yes = []
    pos_attr_no = []
    neg_attr_yes = []
    neg_attr_no = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, data in enumerate(tqdm(f)):
            data = json.loads(data)
            title = data['title']
            # 去除title中年份和数字
            title = ''.join([ch for ch in title if (not ch.isdigit()) and (ch != '年')])
            data['title'] = title
            
            for query in querys:
                values = attr_dict[query]
                if (query == '裤门襟') and ('裤' not in title):
                    continue
                if (query == '闭合方式') and ('鞋' not in title or '靴' not in title):
                    continue
                for value in values:
                    if query == '衣长' and '中长款' in title:
                        data['key_attr'][query] = '中款'
                        break
                    if query == '裙长' and '中长裙' in title:
                        data['key_attr'][query] = '中裙'
                        break
                    if value in title:
                        data['key_attr'][query] =  value

            if data['match']['图文'] == 1: 
                if data['key_attr']:    
                    pos_attr_yes.append(json.dumps(data, ensure_ascii=False)+'\n')
                else:
                    pos_attr_no.append(json.dumps(data, ensure_ascii=False)+'\n')
            else: 
                if data['key_attr']:    
                    neg_attr_yes.append(json.dumps(data, ensure_ascii=False)+'\n')
                else:
                    neg_attr_no.append(json.dumps(data, ensure_ascii=False)+'\n')
    
    print('pos attr yes {:} | pos attr no {:} | neg attr yes {:} | neg attr no {:} |'.format(len(pos_attr_yes), len(pos_attr_no), len(neg_attr_yes), len(neg_attr_no)))
    print('sum is : ', len(pos_attr_yes) + len(pos_attr_no) + len(neg_attr_yes) + len(neg_attr_no))
    return pos_attr_yes, pos_attr_no, neg_attr_yes, neg_attr_no

coarse_path = './data/train_coarse.txt'

coarse_pos_attr_yes, coarse_pos_attr_no, coarse_neg_attr_yes, coarse_neg_attr_no = generate_key_attr(coarse_path)


