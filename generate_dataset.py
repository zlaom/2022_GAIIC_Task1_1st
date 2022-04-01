
import json
from tqdm import tqdm 


# load attribute dict
with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_dict = json.load(f)

test_data_list = []

with open('./data/preliminary_testA.txt', 'r', encoding='utf-8') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        # print(data)
        title = data['title']
        data['title'] = title
        data['key_attr'] = {}
        for query in data['query']:
            if query == '图文':
                continue
            for values in attr_dict[query]:
                flag = False
                for val in values:
                    if val in title:
                        data['key_attr'][query] = val
                        flag = True
                        break
                if flag:
                    break

        test_data_list.append(json.dumps(data, ensure_ascii=False)+'\n')

with open('./data/test_attr_A.txt', 'w', encoding='utf-8') as f:
    f.writelines(test_data_list)
        