from tqdm import tqdm 
import json
import random 

# 加载新的属性字典
attr_file = 'data/new_data/equal_processed_data/attr_to_attrvals.json'
with open(attr_file, 'r') as f:
    attr_dict = json.load(f)

file = 'data/new_data/equal_split_word/nofeat/fine50000.txt'
save_file = 'data/new_data/equal_processed_data/analysis/fine50000.json'

proba_negative_dict = {}
for query, attr_list in attr_dict.items():
    proba_negative_dict[query] = {}
    proba_negative_dict[query]['attr_list'] = attr_list
    proba_list = []
    for attr in attr_list:
        proba_list.append(0)
    proba_negative_dict[query]['attr_freq'] = proba_list


query_list = list(attr_dict.keys())
with open(file, 'r') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        img_name = data['img_name']
        title = data['title']
        key_attr = data['key_attr']

        for query, attr in key_attr.items():
            attr_list = proba_negative_dict[query]['attr_list']
            idx = attr_list.index(attr)
            proba_negative_dict[query]['attr_freq'][idx] += 1
        
with open(save_file, 'w') as f:
    json.dump(proba_negative_dict, f, ensure_ascii=False, indent=4)
        