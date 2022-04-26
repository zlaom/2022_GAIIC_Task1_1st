import os
from tqdm import tqdm 
import json
import itertools

attr_dict_file = "data/contest_data/attr_to_attrvals.json"
fine_file = 'data/contest_data/train_fine.txt'
coarse_file = 'data/contest_data/train_coarse.txt'
testA_file = 'data/contest_data/preliminary_testA.txt'
test_file = 'data/contest_data/preliminary_testB.txt'

save_dir = 'data/tmp_data/equal_processed_data'
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

# 生成id-attr转换字典
attr_id = 0
attr_to_id = {}
id_to_attr = {}
for key, value in attr_dict.items():
    for v in value:
        attr_to_id[v] = attr_id
        id_to_attr[attr_id] = v
        attr_id+=1s
        
attr_to_id_save_file = os.path.join(save_dir, 'attr_to_id.json')
with open(attr_to_id_save_file, 'w') as f:
    json.dump(attr_to_id, f, ensure_ascii=False, indent=4)
id_to_attr_save_file = os.path.join(save_dir, 'id_to_attr.json')
with open(id_to_attr_save_file, 'w') as f:
    json.dump(id_to_attr, f, ensure_ascii=False, indent=4)


# 生成负例字典
import copy
neg_attr_dict = {}
for curr_key, curr_attr_list in attr_dict.items():
    for item_attr in curr_attr_list:
        similar_attr = copy.deepcopy(curr_attr_list)
        similar_attr.remove(item_attr)
        un_similar_attr = []
        for un_similar_key, un_similar_attr_list in attr_dict.items():
            if un_similar_key != curr_key:
                un_similar_attr.extend(un_similar_attr_list)
        neg_attr_dict[item_attr]={
            "similar_attr":similar_attr,
            "un_similar_attr":un_similar_attr
        }

# 保存负例属性字典
neg_attr_save_file = os.path.join(save_dir, 'neg_attr.json')
with open(neg_attr_save_file, 'w') as f:
    json.dump(neg_attr_dict, f, ensure_ascii=False, indent=4)

# exit()
    
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


# [fine] 划分train val数据
print('divide fine data into 45000 and 5000 two parts')
fine_path = os.path.join(save_dir, 'fine50000.txt')
fine_train_path = os.path.join(save_dir, 'fine45000.txt')
fine_val_path = os.path.join(save_dir, 'fine5000.txt')

train_rets = []
val_rets = []

with open(fine_path, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        if len(train_rets) < 45000:      
            train_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        else:
            val_rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
print(len(train_rets))
print(len(val_rets))

with open(fine_train_path, 'w') as f:
    f.writelines(train_rets)
with open(fine_val_path, 'w') as f:
    f.writelines(val_rets)
    
    
# [test] 基础处理同上，唯一的区别是根据query提取key_attr
print('preprocess test A data')
testA_save_file = os.path.join(save_dir, 'test4000.txt')

rets = []
years = ['2017年','2018年','2019年','2020年','2021年','2022年']
equal_list = list(equal_dict.keys())
with open(testA_file, 'r') as f:
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
        for query in data['query']:
            if query != '图文':
                flag = 0
                attr_list = attr_dict[query]
                for attr in attr_list:
                    if attr in title:
                        key_attr[query] = attr  
                        flag = 1
                if flag == 0: # 检查有没有没对应上的query
                    print(data['title'])
                    print(data['query'])
            
        data['key_attr'] = key_attr
        data['title'] = title
        feature = data['feature']
        del data['feature']
        data['feature'] = feature
        
        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
print(len(rets))
with open(testA_save_file, 'w') as f:
    f.writelines(rets)
    
    
# [test] 基础处理同上，唯一的区别是根据query提取key_attr
print('preprocess test B data')
test_save_file = os.path.join(save_dir, 'test10000.txt')

rets = []
years = ['2017年','2018年','2019年','2020年','2021年','2022年']
equal_list = list(equal_dict.keys())
with open(test_file, 'r') as f:
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
        for query in data['query']:
            if query != '图文':
                flag = 0
                attr_list = attr_dict[query]
                for attr in attr_list:
                    if attr in title:
                        key_attr[query] = attr  
                        flag = 1
                if flag == 0: # 检查有没有没对应上的query
                    print(data['title'])
                    print(data['query'])
            
        data['key_attr'] = key_attr
        data['title'] = title
        feature = data['feature']
        del data['feature']
        data['feature'] = feature
        
        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
        
print(len(rets))
with open(test_save_file, 'w') as f:
    f.writelines(rets)


# 生成attr需要的数据划分
print('divide data for attr matching training...')
# [coarse89588 coarse85000 coarse4588]

coarse_file = os.path.join(save_dir, 'coarse89588.txt')
save_coarse_file_train = os.path.join(save_dir, 'coarse85000.txt')
save_coarse_file_val = os.path.join(save_dir, 'coarse4588.txt')

rets_train = []
rets_val = []
for file in coarse_file.split(','):
    with open(file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            item = json.loads(line)
            # del item['feature']
            if len(rets_train) < 85000:
                rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
            elif len(rets_val) < 4588:
                rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        

with open(save_coarse_file_train, 'w') as f:
    f.writelines(rets_train)
with open(save_coarse_file_val, 'w') as f:
    f.writelines(rets_val)