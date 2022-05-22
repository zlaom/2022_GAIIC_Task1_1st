import os
import json


def main():
    # all_path_list = [
    #                  '/home/mw/project/result/fl/task_2_order_loss_0.1649.txt',
    #                  '/home/mw/project/result/fl/task_2_11_loss_0.1477.txt',
    #                  '/home/mw/project/result/fl/task_2_43_loss_0.1452.txt',
    #                  ]

    all_path_list = [
                        '/home/mw/project/result/lhq/title/order_0_5/remake/float/fold0.txt',
                        '/home/mw/project/result/lhq/title/order_0_5/remake/float/fold5.txt',
                        '/home/mw/project/result/lhq/title/order_0_5/remake/float/foldr.txt'
                     ]
    
    os.makedirs('./result', exist_ok=True)
    output_path = '/home/mw/project/result/lhq/title/final_merge/order_0_5.txt'

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

        # if data['match']['图文'] == 1:
        #     for key, val in data['match'].items():
        #         data['match'][key] = 1

        rets.append(json.dumps(data, ensure_ascii=False)+'\n')
    
    print(len(rets))
    print('postive num: ', count)

    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(rets)

main()