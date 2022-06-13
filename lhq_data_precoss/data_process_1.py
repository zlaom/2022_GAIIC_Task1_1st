import os
from tqdm import tqdm 
import json
import itertools
import yaml

print(os.path.abspath(__file__))
yaml_path = '../new_config.yaml'

with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
attr_dict_file = config['data_process']['origin_data']['attr_dict_data']
fine_file = config['data_process']['origin_data']['fine_data']
coarse_file = config['data_process']['origin_data']['coarse_data']


save_dir = config['data_process']['save_data']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# 用来整合相同的属性
equal_dict = {'半高领': '高领',
 '立领': '高领',
 '可脱卸帽': '连帽',
 '衬衫领': '翻领',
 'POLO领': '翻领',
 '方领': '翻领',
 '娃娃领': '翻领',
 '荷叶领': '翻领',
 '五分袖': '短袖',
 '九分袖': '长袖',
 '超短款': '短款',
 '常规款': '短款',
 '超长款': '长款',
 '标准型': '修身型',
 '超短裙': '短裙',
 '中长裙': '中裙', 
 'O型裤': '哈伦裤',
 '灯笼裤': '哈伦裤',
 '锥形裤': '哈伦裤',
 '铅笔裤': '直筒裤',
 '小脚裤': '直筒裤',
 '微喇裤': '喇叭裤',
 '九分裤': '长裤',
 '套筒': '一脚蹬',
 '套脚': '一脚蹬',
 '中帮': '高帮'}

# 加载原始的属性字典
def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict

# load attribute dict
attr_dict = load_attr_dict(attr_dict_file)

# 相等替换
for query, attrs in attr_dict.items():
    attrs = attrs.copy()
    for i, attr in enumerate(attrs):
        if attr in equal_dict:
            attr_dict[query].remove(attr)
            
# 特殊的几个属性替换
for query, attrs in attr_dict.items():
    attrs = attrs.copy()
    for i, attr in enumerate(attrs):
        if query=='衣长' and attr=='中长款':
            attr_dict[query][i] = '中款'
        if query=='裤门襟' and attr=='拉链':
            attr_dict[query][i] = '拉链裤'
        if query=='裤门襟' and attr=='系带':
            attr_dict[query][i] = '系带裤'
        if query=='裤门襟' and attr=='松紧':
            attr_dict[query][i] = '松紧裤'
        if query=='闭合方式' and attr=='拉链':
            attr_dict[query][i] = '拉链鞋'
        if query=='闭合方式' and attr=='系带':
            attr_dict[query][i] = '系带鞋'

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
            # 替换相同属性，fine的替换是从属性反向推回到title的替换
            if attr in equal_dict:
                key_attr[query] = equal_dict[attr]
                # equal_dict的选词很讲究，大多是长词替换成短词，避免了replace可能的出错
                # replace会替换所有满足条件的词，虽然可能都只有一次
                title = title.replace(attr, equal_dict[attr]) 
            # 替换特殊属性
            if query=='衣长' and attr=='中长款':
                key_attr[query] = '中款'
                title = title.replace(attr, '中款')
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
        if '厚度常规' in title:
            title = title.replace('厚度常规', '常规厚度')
        
        data['key_attr'] = key_attr
        data['title'] = title
        # del data['feature']
        
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

equal_list = list(equal_dict.keys())
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
        # 由于替换后的属性不存在包含的情况，用来做属性提取不易出错，所以先做属性替换
        # 相同属性替换
        for attr in equal_list:
            if attr in title:
                title = title.replace(attr, equal_dict[attr])
        # 特殊属性替换
        if '中长款' in title:
            title = title.replace('中长款', '中款')
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
        # 一个高频词的特殊处理
        if '厚度常规' in title:
            title = title.replace('厚度常规', '常规厚度')
        # 属性提取
        if data['match']['图文'] == 1:
            for query in query_list:
                attr_list = attr_dict[query]
                for attr in attr_list:
                    if attr in title:
                        key_attr[query] = attr
                        data['match'][query] = 1   
            
        data['key_attr'] = key_attr
        data['title'] = title
        # del data['feature']
        
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


    


