from ltp import LTP
from tqdm import tqdm
import json

ltp_path = '../../data/pretrained_model/ltp_base'

attr_dict_file = '../../data/equal_processed_data/attr_to_attrvals.json'

fine_file = '../../data/equal_processed_data/fine45000.txt'
coarse_file = '../../data/equal_processed_data/coarse89588.txt'

# fine_file = '../../data/original_data/sample/train_fine_sample.txt'
# coarse_file = '../../data/original_data/sample/train_coarse_sample.txt'

fine_save_file = '../../data/split_word/fine45000.txt'
coarse_save_file = '../../data/split_word/coarse89588.txt'

# fine_save_file = '../../data/split_word/nofeat/fine45000_nofeat.txt'
# coarse_save_file = '../../data/split_word/nofeat/coarse89588_nofeat.txt'

word_dict_save_file = '../vocab/word_dict.json'


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


# 统计训练集的词表和词频，保存分词数据
fine_rets = []
coarse_rets = []
word_dict = {}
with open(fine_file, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item['title']])
        item['title_split'] = segment[0]
        # 重置feature顺序
        feature = item['feature']
        del item['feature']
        item['feature'] = feature
        fine_rets.append(json.dumps(item, ensure_ascii=False)+'\n')
        # 统计词频
        for word in segment[0]:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1


with open(coarse_file, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        segment, _ = ltp.seg([item['title']])
        item['title_split'] = segment[0]
        # 重置feature顺序
        feature = item['feature']
        del item['feature']
        item['feature'] = feature
        coarse_rets.append(json.dumps(item, ensure_ascii=False)+'\n')
        # 统计词频
        for word in segment[0]:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

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