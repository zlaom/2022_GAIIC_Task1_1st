# %%
import os
import json
import joblib
import numpy as np
import random
from tqdm import tqdm 

# %%
# 数据
train_file = 'data/tmp_data/equal_processed_data/fine45000.txt,data/tmp_data/equal_processed_data/coarse85000.txt'
val_file = 'data/tmp_data/equal_processed_data/fine5000.txt,data/tmp_data/equal_processed_data/coarse4588.txt'
neg_attr_dict_file = 'data/tmp_data/equal_processed_data/neg_attr.json'
attr_to_attrvals = 'data/tmp_data/equal_processed_data/attr_to_attrvals.json'
key_attr = "版型"

# %%
with open(attr_to_attrvals, 'r') as f:
    attr_to_attrvals = json.load(f)

# %%
key_attr_values = attr_to_attrvals[key_attr]
id_to_attr = {}
attr_to_id = {}
for attr_id, attr_v in enumerate(key_attr_values):
    attr_to_id[attr_v] = attr_id
    id_to_attr[attr_id] = attr_v

# %%
with open(neg_attr_dict_file, 'r') as f:
    neg_attr_dict = json.load(f)

# %%
# 提取数据
def get_data(file_path, key_attr):
    X1 = []
    X2 = []
    Y = []
    i = 0
    for file in file_path.split(','):
        with open(file, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line)
                if item['match']['图文']: # 训练集图文必须匹配
                    if item['key_attr']: # 必须有属性
                        # 生成所有离散属性
                        for key, attr_value in item['key_attr'].items():
                            # 只提取该类别
                            if key == key_attr:
                                # 删除title节省内存
                                X1.append(item["feature"])
                                X2.append(attr_to_id[attr_value])
                                Y.append(1)
                                i+=1
                                # # 生成同类负例
                                # sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]
                                # attr_value = random.sample(sample_attr_list, k=1)[0]
                                # X1.append(item["feature"])
                                # X2.append(attr_to_id[attr_value])
                                # Y.append(0)
                                # i+=1
                # if i >5000:
                #     break
    print(f"item_num: {i}")
    return np.array(X1), np.array(X2), np.array(Y)



# %%
X_train, Y_train, Y_train2 = get_data(train_file, key_attr)

# %%
X_val, Y_val, Y_val2 = get_data(val_file, key_attr)

# %%
Y_train.shape, Y_val.shape

# %%
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)


val_pre = clf.predict(X_val)
acc = accuracy_score(Y_val, val_pre)


# %%
save_dir = f'data/model_data/svm2048/{key_attr}/'
save_name = f'final_model_{acc:.4f}.pkl'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

joblib.dump(clf, os.path.join(save_dir, save_name)) 