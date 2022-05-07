import os
from tqdm import tqdm
from ltp import LTP 
import json
gpus = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

attr_dict_file = 'data/new_data/equal_processed_data/attr_to_attrvals.json'
test_file = 'data/original_data/preliminary_testA.txt'
ltp_path = 'data/pretrained_model/ltp_base'
vocab_dict_save_file = 'data/new_data/vocab/vocab_dict.json'

SAVE_DIR = 'data/new_data'
test_save_dir = os.path.join(SAVE_DIR, 'test_data')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)
test_save_name = 'test4000.txt' # 决定测试集保存的名字


with open(attr_dict_file, 'r') as f:
    attr_dict = json.load(f)


# ----------------[test] 基础处理,根据query提取key_attr-------------------- #
print('preprocess test data')
save_test_file1 = os.path.join(test_save_dir, 'preprocess_'+test_save_name)

rets = []
years = ['2017年','2018年','2019年','2020年','2021年','2022年']
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
        # 特殊属性替换
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
        if '常规厚度' not in title and '厚度常规款' not in title and '厚度常规' in title:
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
                        break
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
with open(save_test_file1, 'w') as f:
    f.writelines(rets)

# ----------------------------分词------------------------------------- #
print('split words for testing data:')
print('split words for test data...')

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
extra_words.append(['加厚', '薄款', '厚款', '短外套', '常规厚度'])
extra_words.append(['不加绒', '无扣', '无弹力', '无弹', '无拉链'])
extra_words.append(['一粒扣', '两粒扣', '暗扣', '三粒扣', '系扣'])
extra_words.append(['大红色', '大花'])
for extra in extra_words:   
    all_attr = all_attr + extra

ltp.init_dict(path="user_dict.txt", max_window=6)
ltp.add_words(words=all_attr, max_window=6)


save_test_file2 = os.path.join(test_save_dir, 'split_word_'+test_save_name)
rets = []
with open(save_test_file1, 'r') as f:
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
print(len(rets))
with open(save_test_file2, 'w') as f:
    f.writelines(rets)
    
    
# ----------------------------分词后处理------------------------------------ #
with open(vocab_dict_save_file, 'r') as f:
    vocab_dict = json.load(f)
    
vocab_list = list(vocab_dict.keys())
color_list = ['兰','蓝','灰','绿','粉','红','黄','青','紫','白','黑','驼','橙','杏','咖','棕','啡','褐','银','金','橘','藏']
mul_color_list = ['卡其', '咖啡']

print('processing test split-word data...')
save_test_file3 = os.path.join(test_save_dir, 'equal_split_word_'+test_save_name)
rets = []
with open(save_test_file2, 'r') as f:
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
        
with open(save_test_file3, 'w') as f:
    f.writelines(rets)