import os
import json


def main():
    result_dir = 'project/submmision'
    all_path_list = [
                        os.path.join(result_dir, 'fold0.txt'),
                        os.path.join(result_dir, 'fold3.txt'),
                        os.path.join(result_dir, 'fold5.txt'),
                        os.path.join(result_dir, 'order.txt')
                     ]
    output_path = os.path.join(result_dir, 'title_merge.txt')

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
            if val > k_fold / 2.0:
                data['match'][key] = 1
                if key == '图文':
                    count += 1
            else:
                data['match'][key] = 0

        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
    
    print(len(rets))
    print('postive num: ', count)

    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)

main()