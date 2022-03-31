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
├── attr_to_attrvals.json
├── test.txt
├── train_coarse.txt
└── train_fine.txt
```
分别为属性字典文件、测试文件、粗标数据、细标数据。

划分数据：
```
split -l 49000 -d data/train_fine.txt data/train_fine.txt.
```
`data/train_fine.txt.01`用于模型验证。

使用如下脚本训练模型：
```
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export TOKENIZERS_PARALLELISM=false

python -u src/training/main.py \
    --save-frequency 1     \
    --train-data="data/train_coarse.txt,data/train_fine.txt.00"    \
    --val-data="data/train_fine.txt.01"   \
    --dataset-type="json" \
    --warmup 1000  \
    --batch-size=128  \
    --lr=1e-4  \
    --wd=0.1  \
    --epochs=100   \
    --workers=4   \
    --dp   \
    --multigpu 0,1,2,3 \
    --name "demo"
```
如果出现OOM错误，可以尝试降低batch-size。

## 测试
demo.py演示使用训练好的模型进行图文、属性匹配的预测：
```
export PYTHONPATH="$PYTHONPATH:$PWD/src"
python demo.py
```
test.txt对应的预测结果保存到test_pred.txt。
