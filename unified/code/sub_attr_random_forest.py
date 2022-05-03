# %%
import os
import json
import joblib
import argparse 
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser('train_attr', add_help=False)
parser.add_argument('--index', default='0', type=int)
args = parser.parse_args()   

# %%
# 数据
train_file = 'data/tmp_data/equal_processed_data/fine45000.txt,data/tmp_data/equal_processed_data/coarse85000.txt'
val_file = 'data/tmp_data/equal_processed_data/fine5000.txt,data/tmp_data/equal_processed_data/coarse4588.txt'
neg_attr_dict_file = 'data/tmp_data/equal_processed_data/neg_attr.json'
attr_to_attrvals = 'data/tmp_data/equal_processed_data/attr_to_attrvals.json'
key_attr_list = [['领型', '袖长', '衣长'], ['版型', '裙长', '穿着方式'], ['类别', '裤型', '裤长'], ['裤门襟', '闭合方式', '鞋帮高度']]
current_key_attr = key_attr_list[args.index]

# %%
with open(attr_to_attrvals, 'r') as f:
    attr_to_attrvals = json.load(f)

for key_attr in current_key_attr:
    print(f"begin train {key_attr}")
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
                                    # 生成同类负例
                                    # sample_attr_list = neg_attr_dict[attr_value]["similar_attr"]
                                    # attr_value = random.sample(sample_attr_list, k=1)[0]
                                    # X1.append(item["feature"])
                                    # X2.append(attr_to_id[attr_value])
                                    # Y.append(0)
                                    # i+=1
                    # if i >1000:
                    #     break
        print(f"item_num: {i}")
        return np.array(X1), np.array(X2), np.array(Y)


    # %%
    X_train, L_train, Y_train = get_data(train_file, key_attr)

    # %%
    X_val, L_val, Y_val = get_data(val_file, key_attr)

    # %%
    print(X_train.shape, L_train.shape, Y_train.shape)

    #%%
    # from sklearn.preprocessing import OneHotEncoder
    # id_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)


    # id_num = len(key_attr_values)
    # id_encoder.fit(np.arange(id_num).reshape(-1, 1))

    # L_train = id_encoder.transform(L_train.reshape(-1, 1))
    # L_val = id_encoder.transform(L_val.reshape(-1, 1))

    # XL_train = np.concatenate((X_train, L_train), axis=-1)
    # XL_val = np.concatenate((X_val, L_val), axis=-1)

    # %%
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, L_train)


    val_pre = clf.predict(X_val)
    acc = accuracy_score(L_val, val_pre)


    # %%
    save_dir = f'data/model_data/svm_sub_attr/{key_attr}/'
    save_name = f'final_model_{acc:.4f}.pkl'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    joblib.dump(clf, os.path.join(save_dir, save_name)) 