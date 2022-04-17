import os
import json


def main():

    all_path_list = [
                     './result/test_A_split_no_0.9445_attr_0.9339.txt',
                     './result/test_A_split_no_0.9570_attr_0.9300.txt',
                     './result/pred_title0.9398_attr0.9356.txt',
                     ]

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

        if data['match']['图文'] == 1:
            for key, val in data['match'].items():
                data['match'][key] = 1

        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
    
    print(len(rets))
    print('postive num: ', count)

    os.makedirs('./result', exist_ok=True)
    output_path = './result/NEW_SE_0.9400_' + str(k_fold) + '_match.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)

main()