import os
import json


def main():

    all_path_list = ['./result/split1_attr_0.932.txt',
                     './result/split2_attr_0.932.txt',
                     './result/split3_attr_0.932.txt',
                     './result/split4_attr_0.932.txt',
                     './result/split5_attr_0.932.txt',
                     './result/split6_attr_0.932.txt',
                     './result/split7_attr_0.932.txt',
                     './result/split8_attr_0.932.txt',
                     './result/split9_attr_0.932.txt']

    k_fold = len(all_path_list)
    k_fold_data_list = []
    for path in all_path_list:
        per_all_data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                per_all_data.append(data)
        k_fold_data_list.append(per_all_data)
    
    rets = []
    count = 0
    for i in range(len(k_fold_data_list[0])):
        for j in range(k_fold):
            if j == 0:
                data = k_fold_data_list[0][i]
            else:
                for key in data['match'].keys():
                    data['match'][key] += k_fold_data_list[j][i]['match'][key]
        
        for key, val in data['match'].items():
            if val > k_fold // 2:
                data['match'][key] = 1
                if key == '图文':
                    count += 1
            else:
                data['match'][key] = 0

        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
    
    print(len(rets))
    print('postive num: ', count)

    os.makedirs('./result', exist_ok=True)
    output_path = './result/k_fold_' + str(k_fold) + '_attr_0.932.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)

main()