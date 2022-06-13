import os
from tqdm import tqdm 
import json
import itertools
import yaml

print(os.path.abspath(__file__))
yaml_path = '../new_config_1.yaml'

with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
attr_dict_file = config['data_process_1']['origin_data']['attr_dict_data']
fine_file = config['data_process_1']['origin_data']['fine_data']
coarse_file = config['data_process_1']['origin_data']['coarse_data']


save_dir = config['data_process_1']['save_data']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

# 加载原始的属性字典
def load_attr_dict(file):
    # 读取属性字典
    i = 0
    with open(file, 'r') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            tmp = []
            for vals in attrval_list:
                if '=' in vals:
                    tmp.append(vals.split('='))
                    i += len(vals.split('='))
                else:
                    tmp.append([vals])
                    i += len([vals])
            attr_dict[attr] = tmp
    print('number of the attr : ', i)
    return attr_dict

# load attribute dict
attr_dict = load_attr_dict(attr_dict_file)


# 包含替换
for query, all_attrs in attr_dict.items():
    all_attrs = all_attrs.copy()
    for j, attrs in enumerate(all_attrs):
        for i, attr in enumerate(attrs):
            # 相同query的包含替换
            if query=='领型' and attr=='半高领':
                attr_dict[query][j][i] = '高半领'
            if query=='衣长' and attr=='超短款':
                attr_dict[query][j][i] = '短超款'
            if query=='衣长' and attr=='超长款':
                attr_dict[query][j][i] = '长超款'
            if query=='衣长' and attr=='中长款':
                attr_dict[query][j][i] = '长中款'
            if query=='裙长' and attr=='超短裙':
                attr_dict[query][j][i] = '短超裙'
            if query=='裙长' and attr=='中长裙':
                attr_dict[query][j][i] = '中大裙'
            
            # 不同query的包含替换
            if query=='裤门襟' and attr=='拉链':
                attr_dict[query][j][i] = '拉链裤'
            if query=='裤门襟' and attr=='系带':
                attr_dict[query][j][i] = '系带裤'
            if query=='裤门襟' and attr=='松紧':
                attr_dict[query][j][i] = '松紧裤'
            if query=='闭合方式' and attr=='拉链':
                attr_dict[query][j][i] = '拉链鞋'
            if query=='闭合方式' and attr=='系带':
                attr_dict[query][j][i] = '系带鞋'


# 保存新的属性字典
attr_save_file = os.path.join(save_dir, 'attr_to_attrvals.json')
with open(attr_save_file, 'w') as f:
    json.dump(attr_dict, f, ensure_ascii=False, indent=4)
    
    
# [fine] 移除年份，统一大写，替换相等属性，替换特殊属性
print('preprocess fine data')
new_fine_file = os.path.join(save_dir, 'fine50000.txt')
rets = []
years = ['2017年','2018年','2019年','2020年','2021年','2022年']

with open(fine_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        title = data['title']
        key_attr = data['key_attr']
        # 删除年份
        for year in years:
            title = title.replace(year, '')
        # 统一大写
        title = title.upper() # 字母统一为大写
        # 属性替换
        for query, attr in key_attr.items():
            # 替换特殊属性
            # 相同query的包含替换
            if query=='领型' and attr=='半高领':
                key_attr[query] = '高半领'
                title = title.replace(attr, '高半领')
            if query=='衣长' and attr=='超短款':
                key_attr[query] = '短超款'
                title = title.replace(attr, '短超款')
            if query=='衣长' and attr=='超长款':
                key_attr[query] = '长超款'
                title = title.replace(attr, '长超款')
            if query=='衣长' and attr=='中长款':
                key_attr[query] = '长中款'
                title = title.replace(attr, '长中款')
            if query=='裙长' and attr=='超短裙':
                key_attr[query] = '短超裙'
                title = title.replace(attr, '短超裙')
            if query=='裙长' and attr=='中长裙':
                key_attr[query] = '中大裙'
                title = title.replace(attr, '中大裙')

            # 不同query的包含替换
            if query=='裤门襟' and attr=='拉链' and '无拉链' not in title:
                key_attr[query] = '拉链裤'
                title = title.replace(attr, '拉链裤')
            if query=='裤门襟' and attr=='系带':
                key_attr[query] = '系带裤'
                title = title.replace(attr, '系带裤')
            if query=='裤门襟' and attr=='松紧':
                key_attr[query] = '松紧裤'
                title = title.replace(attr, '松紧裤')
            if query=='闭合方式' and attr=='拉链':
                key_attr[query] = '拉链鞋'
                title = title.replace(attr, '拉链鞋')
            if query=='闭合方式' and attr=='系带':
                key_attr[query] = '系带鞋'
                title = title.replace(attr, '系带鞋')
        # 一个高频词的特殊处理
        if '常规厚度' not in title and '厚度常规款' not in title and '厚度常规' in title:
            title = title.replace('厚度常规', '常规厚度')
        
        data['key_attr'] = key_attr
        data['title'] = title
        
        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
with open(new_fine_file, 'w') as f:
    f.writelines(rets)
    
# [coarse] 移除年份，统一大写，替换相等属性，替换特殊属性
print('preprocess coarse data')
pos_coarse_file = os.path.join(save_dir, 'coarse89588.txt')
neg_coarse_file = os.path.join(save_dir, 'coarse10412.txt')

pos_rets = []
neg_rets = []
years = ['2017年','2018年','2019年','2020年','2021年','2022年']

query_list = list(attr_dict.keys()) # 注意是新属性字典
with open(coarse_file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        title = data['title']
        key_attr = {}
        # 删除年份
        for year in years:
            title = title.replace(year, '')
        # 统一大写
        title = title.upper() # 字母统一为大写
        # 相同query的包含替换
        if '半高领' in title:
            title = title.replace('半高领', '高半领')
        if '超短款' in title:
            title = title.replace('超短款', '短超款')
        if '超长款' in title:
            title = title.replace('超长款', '长超款')
        if '中长款' in title:
            title = title.replace('中长款', '长中款')
        if '超短裙' in title:
            title = title.replace('超短裙', '短超裙')
        if '中长裙' in title:
            title = title.replace('中长裙', '中大裙')
        
        # 不同query的包含替换
        if '拉链' in title and '裤' in title and '无拉链' not in title:
            title = title.replace('拉链', '拉链裤')
        if '系带' in title and '裤' in title:
            title = title.replace('系带', '系带裤')
        if '松紧' in title and '裤' in title:
            title = title.replace('松紧', '松紧裤')
        if '拉链' in title and ('鞋' in title or '靴' in title):
            title = title.replace('拉链', '拉链鞋')
        if '系带' in title and ('鞋' in title or '靴' in title):
            title = title.replace('系带', '系带鞋')
        
        # 属性提取
        if data['match']['图文'] == 1:
            for query in query_list:
                attr_list = attr_dict[query]
                for attrs in attr_list:
                    for attr in attrs:
                        if attr in title:
                            key_attr[query] = attr
                            data['match'][query] = 1 
        # 一个高频词的特殊处理
        if '常规厚度' not in title and '厚度常规款' not in title and '厚度常规' in title:
            title = title.replace('厚度常规', '常规厚度')                  
        data['key_attr'] = key_attr
        data['title'] = title
        
        if data['match']['图文'] == 1:
            pos_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        else:
            neg_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
print(len(pos_rets))
print(len(neg_rets))
with open(pos_coarse_file, 'w') as f:
    f.writelines(pos_rets)
with open(neg_coarse_file, 'w') as f:
    f.writelines(neg_rets)


    


