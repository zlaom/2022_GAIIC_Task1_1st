# 说明

2022世界人工智能大赛赛道一-电商图文匹配
sustech小分队 baseline

## 训练
首先安装open_clip（参考[README_openclip.md](README_openclip.md)），再安装其他依赖：
```
pip install transformers
```
注意，如果环境中安装了clip包，需要卸载，否则训练时可能出错。

下载数据，放到data文件夹下：
```
data
├── MacBert
├── attr_match.json
├── attr_to_attrvals.json
├── finetune_all_match.txt
├── neg_coarse.txt
├── pos_coarse_attr.txt
├── preliminary_testA.txt
├── test_attr_A.txt
├── train_all_match.txt
├── train_coarse.txt
└── train_fine.txt
```
分别为属性字典文件、测试文件、粗标数据、细标数据。


生成60w+的属性正负例, 29w的图文正负例：
```
process_data.ipynb
```

生成test的属性值匹配：
```
python generate_dataset.py
```

属性训练
```
python train_attr_match.py
```

图文训练+finetune
```
python train_all_match.py
python finetune.py
```

测试
```
python test_2.py
```

