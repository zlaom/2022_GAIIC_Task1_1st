import os 
import numpy as np
import json

data_path_1 = './data/train_fine.txt'
data_path_2 = './data/train_coarse.txt'

data_pos_list = []
data_neg_list = []

with open(data_path_1, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data['match']['图文'] == 1:
            data_pos_list.append(data)
        else:
            data_neg_list.append(data)

with open(data_path_2, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data['match']['图文'] == 1:
            data_pos_list.append(data)
        else:
            data_neg_list.append(data)

np.random.shuffle(data_pos_list)
x_train_list = data_pos_list[:len(data_pos_list)-len(data_neg_list)]
x_val_list = data_pos_list[len(data_pos_list)-len(data_neg_list):] + data_neg_list
np.random.shuffle(x_val_list)

pre_ret = []
train_ret = []
val_ret = []
for dic in x_train_list:
    pre_ret.append(json.dumps(dic, ensure_ascii=False)+'\n')

l = len(x_val_list)-1000
for i in range(l):
    dic = x_val_list[i]
    train_ret.append(json.dumps(dic, ensure_ascii=False)+'\n')

for i in range(l, len(x_val_list)):
    dic = x_val_list[i]
    val_ret.append(json.dumps(dic, ensure_ascii=False)+'\n')

with open('./data/pretrain_match.txt', 'w', encoding='utf-8') as f:
    f.writelines(pre_ret)

with open('./data/train_match.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_ret)

with open('./data/val_match.txt', 'w', encoding='utf-8') as f:
    f.writelines(val_ret)

print(len(pre_ret))
print(len(train_ret))
print(len(val_ret))
print(len(pre_ret) + len(train_ret) + len(val_ret))