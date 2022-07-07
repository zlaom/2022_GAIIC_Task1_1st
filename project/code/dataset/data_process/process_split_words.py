import os 
from tqdm import tqdm
import json

SAVE_DIR = 'temp/tmp_data/lhq_data'

preprocess_dir = os.path.join(SAVE_DIR, 'equal_processed_data')
split_word_dir = os.path.join(SAVE_DIR, 'split_word')

equal_split_word_dir = os.path.join(SAVE_DIR, 'equal_split_word')
vocab_dir = os.path.join(SAVE_DIR, 'vocab') # 注意要修改后面的保存文件夹位置

if not os.path.exists(equal_split_word_dir):
    os.makedirs(equal_split_word_dir)

# 两个已有的文件
attr_dict_file = os.path.join(preprocess_dir, 'attr_to_attrvals.json')
word_dict_file = os.path.join(vocab_dir, 'word_dict.json')

# 两个要保存的文件
vocab_dict_save_file = os.path.join(vocab_dir, 'vocab_dict.json')
vocab_txt_save_file = os.path.join(vocab_dir, 'vocab.txt')


name_list = ['fine50000.txt', 'coarse89588.txt', 'coarse10412.txt']

GATE = 50
# ---------------------生成vocab_dict.json,经过了词频筛选,组合等步骤--------------------- #
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
    if word_dict[word] < GATE: # 注意这个值要与后面删除的阈值保持一致
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
    if word_dict[word] < GATE and word not in ignore_list:
        del word_dict[word]
        
# 去掉没有单独意义的字词
delete_words = ['色','小','本','中','新','款','加','底','件','不', '/']
for word in delete_words:
    if word in word_dict:
        del word_dict[word]
    
# 保存处理后的词表
vocab_dict = word_dict
with open(vocab_dict_save_file, 'w') as f:
    json.dump(vocab_dict, f, ensure_ascii=False)


# -------------------------生成vocab.txt--------------------------- #
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
        

# -----------------根据处理后的词表对数据的分词进行筛选---------------------- #
vocab_list = list(vocab_dict.keys())
color_list = ['兰','蓝','灰','绿','粉','红','黄','青','紫','白','黑','驼','橙','杏','咖','棕','啡','褐','银','金','橘','藏']
mul_color_list = ['卡其', '咖啡']
for name in name_list:
    print('processing '+name+' split-word data...')
    file = os.path.join(split_word_dir, name)
    save_file = os.path.join(equal_split_word_dir, name)
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
        
        
        
        



