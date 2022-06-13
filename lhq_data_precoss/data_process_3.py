import os 
from tqdm import tqdm
import json
import yaml

yaml_path = '../new_config.yaml'
with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config = config['data_processed_3']

attr_dict_file = config['attr_dict_file']
word_dict_file = config['word_dict_file']
vocab_dict_save_file = config['vocab_dict_save_file']
vocab_txt_save_file = config['vocab_txt_save_file']

data_dir = config['data_dir']
save_dir = config['save_dir']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
name_list = ['coarse89588.txt', 'fine50000.txt', 'coarse10412.txt']

    
# 处理词表
print('generate vocab_dict.json')
# 读入未处理的词表
with open(word_dict_file, 'r') as f:
    word_dict = json.load(f)
    
# 统一大量颜色的别称
color_list = ['兰','蓝','灰','绿','粉','红','黄','青','紫','白','黑','驼','橙','杏','咖','棕','啡','褐','银','金','橘','藏']
word_list = list(word_dict.keys())

def union_word_dict(word_dict, new_word, ori_word):
    if new_word in word_dict:
        word_dict[new_word] += word_dict[ori_word]
    else:
        word_dict[new_word] = word_dict[ori_word]

for word in word_list:
    if word_dict[word] < 50: # 注意这个值要与后面删除的阈值保持一致
        # 两个特殊的颜色
        rep_word = word
        if '卡其' in rep_word:
            union_word_dict(word_dict, '卡其', word)
            rep_word = rep_word.replace('卡其', '')
        if '咖啡' in rep_word:
            union_word_dict(word_dict, '咖啡', word)
            rep_word = rep_word.replace('咖啡', '')
        for char in rep_word:
            if char in color_list:
                union_word_dict(word_dict, char, word)
                
# 读入属性词表
def get_dict(file):
    with open(file, 'r') as f:
        all_attr = []
        for attr, attrval_list in json.load(f).items():
            for x in attrval_list:
                all_attr.append(x)
    return all_attr
all_attr = get_dict(attr_dict_file)

# 删除出现次数少的词，不删除属性词，不删除我们设置的颜色值（否则后面可能会不匹配出现bug）
ignore_list = all_attr + color_list + ['卡其', '咖啡']
word_list = list(word_dict.keys())
for word in word_list:
    if word_dict[word] < 50 and word not in ignore_list:
        del word_dict[word]
    if word == '/':
        del word_dict[word]
        
# 去掉没有单独意义的字词
delete_words = ['色','小','本','中','新','款','加','底','件','不']
for word in delete_words:
    del word_dict[word]
    
# 保存处理后的词表
vocab_dict = word_dict
with open(vocab_dict_save_file, 'w') as f:
    json.dump(vocab_dict, f, ensure_ascii=False)


# 生成bert需要的vocab.txt
print('generate vocab.txt')
# 定义词表
l_vocab = []

# 添加[PAD],10个[unused],[UNK],[CLS],[SEP],[MASK]
l_vocab.append('[PAD]')
for i in range(10):
    l_vocab.append('[unused'+str(i+1)+']')
    
extra_words = ['[UNK]', '[CLS]', '[SEP]', '[MASK]']
l_vocab = l_vocab + extra_words

# 将词表转换为list，并进行排序
l_words = []
for word, num in vocab_dict.items():
    l_words.append(word)
l_words.sort() 
l_vocab = l_vocab + l_words

# 保存为vocab
with open(vocab_txt_save_file, 'w') as writer:
    for word in l_vocab:
        writer.write(word+'\n')
        

# 根据处理后的词表对数据的分词进行筛选
vocab_list = list(vocab_dict.keys())
color_list = ['兰','蓝','灰','绿','粉','红','黄','青','紫','白','黑','驼','橙','杏','咖','棕','啡','褐','银','金','橘','藏']
mul_color_list = ['卡其', '咖啡']
for name in name_list:
    print('processing '+name+' split-word data...')
    file = os.path.join(data_dir, name)
    save_file = os.path.join(save_dir, name)
    rets = []
    with open(file, 'r') as f:
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
        
        
        
        
# # 生成title需要的数据划分
# print('divide data for title matching training...')
# # [fine40000, fine9000, fine700]
# title_dir = os.path.join(save_dir, 'title')
# if not os.path.exists(title_dir):
#     os.makedirs(title_dir)
# fine_file = os.path.join(save_dir, 'fine45000.txt') + ',' + os.path.join(save_dir, 'fine5000.txt')
# save_fine_file_pretrain = os.path.join(title_dir, 'fine40000.txt')
# save_fine_file_train = os.path.join(title_dir, 'fine9000.txt')
# save_fine_file_val = os.path.join(title_dir, 'fine700.txt')

# rets_pretrain = []
# rets_train = []
# rets_val = []
# for file in fine_file.split(','):
#     with open(file, 'r') as f:
#         for i, line in enumerate(tqdm(f)):
#             item = json.loads(line)
#             # del item['feature']
#             if len(rets_pretrain) < 40000:
#                 rets_pretrain.append(json.dumps(item, ensure_ascii=False)+'\n')
#             elif len(rets_train) < 9000:
#                 rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
#             elif len(rets_val) < 700:
#                 rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        
# with open(save_fine_file_pretrain, 'w') as f:
#     f.writelines(rets_pretrain)
# with open(save_fine_file_train, 'w') as f:
#     f.writelines(rets_train)
# with open(save_fine_file_val, 'w') as f:
#     f.writelines(rets_val)


# # [coarse89588 coarse9000 coarse1412]
# coarse_file = os.path.join(save_dir, 'coarse10412.txt')
# save_coarse_file_train = os.path.join(title_dir, 'coarse9000.txt')
# save_coarse_file_val = os.path.join(title_dir, 'coarse1412.txt')

# rets_train = []
# rets_val = []
# for file in coarse_file.split(','):
#     with open(file, 'r') as f:
#         for i, line in enumerate(tqdm(f)):
#             item = json.loads(line)
#             # del item['feature']
#             if len(rets_train) < 9000:
#                 rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
#             elif len(rets_val) < 1412:
#                 rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        

# with open(save_coarse_file_train, 'w') as f:
#     f.writelines(rets_train)
# with open(save_coarse_file_val, 'w') as f:
#     f.writelines(rets_val)



# # 生成attr需要的数据划分
# print('divide data for attr matching training...')
# # [coarse89588 coarse85000 coarse4588]
# attr_dir = os.path.join(save_dir, 'attr')
# if not os.path.exists(attr_dir):
#     os.makedirs(attr_dir)

# coarse_file = os.path.join(save_dir, 'coarse89588.txt')
# save_coarse_file_train = os.path.join(attr_dir, 'coarse85000.txt')
# save_coarse_file_val = os.path.join(attr_dir, 'coarse4588.txt')

# rets_train = []
# rets_val = []
# for file in coarse_file.split(','):
#     with open(file, 'r') as f:
#         for i, line in enumerate(tqdm(f)):
#             item = json.loads(line)
#             # del item['feature']
#             if len(rets_train) < 85000:
#                 rets_train.append(json.dumps(item, ensure_ascii=False)+'\n')
#             elif len(rets_val) < 4588:
#                 rets_val.append(json.dumps(item, ensure_ascii=False)+'\n')
        

# with open(save_coarse_file_train, 'w') as f:
#     f.writelines(rets_train)
# with open(save_coarse_file_val, 'w') as f:
#     f.writelines(rets_val)