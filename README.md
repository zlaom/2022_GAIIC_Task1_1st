# 说明

此代码为JD图文匹配比赛的demo代码，基于[open_clip](https://github.com/mlfoundations/open_clip)和[Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP)开发。


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
├── attr_to_attrvals.json
├── test.txt
├── train_coarse.txt
├── pretrain_match.txt
├── train_math.txt
├── val_math.txt
└── train_fine.txt
└── neg_fine.txt
```
分别为属性字典文件、测试文件、粗标数据、细标数据。


生成5w的负例：
```
analyze.ipynb
```

训练
```
python train.py
```

测试
```
python test.py
```

