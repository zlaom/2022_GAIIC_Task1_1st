import os
from tqdm import tqdm 
import json
import itertools
from ltp import LTP

test_file = '../data/preliminary_testA.txt'
test_save_file_1 = '../data/equal_processed_data/test_A.txt'
test_save_file_2 = '../data/split_word/test_A.txt'
test_save_file_3 = '../data/equal_split_word/test_A.txt'
ltp_path = '../data/ltp_base'
vocab_dict_path = '../data/split_word/vocab/vocab_dict.json'

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

def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict

# load attribute dict
attr_dict_file = '../data/equal_processed_data/attr_to_attrvals.json'
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
# equal_processed_d
with open(test_save_file_1, 'w') as f:
    f.writelines(rets)


ltp = LTP(path=ltp_path)
# 取得新的属性字典并添加到词表
def get_dict(file):
    with open(file, 'r') as f:
        all_attr = []
        for attr, attrval_list in json.load(f).items():
            for x in attrval_list:
                all_attr.append(x)
    return all_attr
all_attr = get_dict(attr_dict_file)

# ltp设置
extra_words = []
extra_words.append(['牛津布', '仿皮', '吸湿', '吸汗', '防滑', '抗冲击', '微弹', '加绒'])
extra_words.append(['上青', '上青色', '上青绿', '羊绒衫'])
extra_words.append(['休闲鞋', '工装鞋', '男包', '女包', '运动裤', '休闲裤', '加厚领'])
extra_words.append(['加厚', '薄款', '厚款', '短款', '短外套', '常规厚度'])
extra_words.append(['不加绒', '无扣', '无弹力', '无弹', '无拉链'])
extra_words.append(['一粒扣', '两粒扣', '暗扣', '三粒扣', '系扣'])
extra_words.append(['大红色', '大花'])
for extra in extra_words:   
    all_attr = all_attr + extra

ltp.init_dict(path="user_dict.txt", max_window=6)
ltp.add_words(words=all_attr, max_window=6)

rets = []
with open(test_save_file_1, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item['title']])
        item['title_split'] = segment[0]
        # 重置feature顺序
        feature = item['feature']
        del item['feature']
        item['feature'] = feature
        rets.append(json.dumps(item, ensure_ascii=False)+'\n')

# 保存数据
# equal_split_word, 分词
print(len(rets))
with open(test_save_file_2, 'w') as f:
    f.writelines(rets)

with open(vocab_dict_path, 'r') as f:
    vocab_dict = json.load(f)


vocab_list = list(vocab_dict.keys())
color_list = ['兰','蓝','灰','绿','粉','红','黄','青','紫','白','黑','驼','橙','杏','咖','棕','啡','褐','银','金','橘','藏']
mul_color_list = ['卡其', '咖啡']
save_file = test_save_file_3
rets = []
with open(test_save_file_2, 'r') as f:
    for i, line in enumerate(tqdm(f)):
        item = json.loads(line)
        title_split = item['title_split']
        
        vocab_split = []
        for word in title_split:
            if word in vocab_list:
                vocab_split.append(word)
            else: # 颜色提取
                rep_word = word
                for mul_color in mul_color_list:
                    if mul_color in rep_word:
                        vocab_split.append(mul_color)
                        rep_word = rep_word.replace(mul_color, '')
                for char in rep_word:
                    if char in color_list:
                        vocab_split.append(char)
        item['vocab_split'] = vocab_split
        # 看看有没有空的vocab_split
        if not vocab_split:
            print(item['title'])
        # 更改保存的顺序，便于查看
        feature = item['feature']
        del item['feature']
        item['feature'] = feature
        
        rets.append(json.dumps(item, ensure_ascii=False)+'\n')
        
with open(save_file, 'w') as f:
    f.writelines(rets)


