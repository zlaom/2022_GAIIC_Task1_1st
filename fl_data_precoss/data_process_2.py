import os
from ltp import LTP
from tqdm import tqdm
import json
import yaml


yaml_path = '../new_config_1.yaml'
with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config = config['data_processed_2']

attr_dict_file = config['attr_dict_file']
fine_file = config['fine_val_file']
coarse_file = config['coarse_file']

coarse_val_file = config['coarse_val_file']

ltp_path = config['ltp_path']


save_dir = config['save_dir']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fine_save_file = os.path.join(save_dir, 'fine50000.txt')
coarse_save_file = os.path.join(save_dir, 'coarse89588.txt')
coarse_val_save_file = os.path.join(save_dir, 'coarse10412.txt')


vocab_dir = config['vocab_dir']
if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)
word_dict_save_file = os.path.join(vocab_dir, 'word_dict.json')


ltp = LTP(path=ltp_path) 

# 取得新的属性字典并添加到词表
def get_dict(file):
    with open(file, 'r') as f:
        all_attr = []
        for query, attrval_list in json.load(f).items():
            for attrs in attrval_list:
                for x in attrs:
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

def get_split_word(path, mode='fine'):
    if mode == 'fine':
        print('split words for fine 50000 data...')
    else:
        print('split words for coarse 89588 data...')
    
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            segment, _ = ltp.seg([item['title']])
            item['title_split'] = segment[0]
            # 重置feature顺序
            feature = item['feature']
            del item['feature']
            item['feature'] = feature
            if mode == 'fine':
                fine_rets.append(json.dumps(item, ensure_ascii=False)+'\n')
            else:
                coarse_rets.append(json.dumps(item, ensure_ascii=False)+'\n')
            # 统计词频
            for word in segment[0]:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
# 训练集分词
# 统计训练集的词表和词频，保存分词数据
print('split words for training data:')
fine_rets = []
coarse_rets = []
word_dict = {}
get_split_word(fine_file)
get_split_word(coarse_file, mode='coarse')

# 保存词表
with open(word_dict_save_file, 'w') as f:
    json.dump(word_dict, f, ensure_ascii=False)
    
# 保存数据
print(len(fine_rets))
print(len(coarse_rets))
with open(fine_save_file, 'w') as f:
    f.writelines(fine_rets)
with open(coarse_save_file, 'w') as f:
    f.writelines(coarse_rets)
    
    
# 验证集分词
# 统计训练集的词表和词频，保存分词数据
with open(word_dict_save_file, 'r') as f:
    word_dict = json.load(f)
coarse_rets = []

print('split words for coarse 10412 data...')
with open(coarse_val_file, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item['title']])
        item['title_split'] = segment[0]
        # 重置feature顺序
        feature = item['feature']
        del item['feature']
        item['feature'] = feature
        coarse_rets.append(json.dumps(item, ensure_ascii=False)+'\n')

print(len(coarse_rets))
with open(coarse_val_save_file, 'w') as f:
    f.writelines(coarse_rets)



