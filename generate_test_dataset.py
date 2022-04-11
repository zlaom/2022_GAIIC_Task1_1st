
import json
from tqdm import tqdm 


# load attribute dict
with open('./data/attr_match.json', 'r', encoding='utf-8') as f:
    attr_dict = json.load(f)

with open('./data/attr_dic_all_1.json', 'r', encoding='utf-8') as f:
    attr_all_dict = json.load(f)

test_data_list = []

with open('./data/preliminary_testA.txt', 'r', encoding='utf-8') as f:
    for i, data in enumerate(tqdm(f)):
        data = json.loads(data)
        # print(data)
        title = data['title']
        title = ''.join([ch for ch in title if (not ch.isdigit()) and (ch != '年')])
        data['title'] = title
        data['key_attr'] = {}
        for query in data['query']:
            if query == '图文':
                continue
            for values in attr_dict[query]:
                flag = False
                for val in values:
                    if query == '衣长' and '中长款' in title:
                        data['key_attr'][query] = '中长款'
                        flag = True
                        break

                    if query == '裙长' and '中长裙' in title:
                        data['key_attr'][query] = '中裙'
                        flag = True
                        break

                    if val in title:
                        for all_v, new_val in attr_all_dict[query].items():
                            if val in all_v:
                                data['key_attr'][query] = new_val
                                flag = True
                                break
                if flag:
                    break

        test_data_list.append(json.dumps(data, ensure_ascii=False)+'\n')

with open('./data/test_attr_A_54.txt', 'w', encoding='utf-8') as f:
    f.writelines(test_data_list)
        