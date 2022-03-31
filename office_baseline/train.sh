export PYTHONPATH="$PYTHONPATH:$PWD/src"
export TOKENIZERS_PARALLELISM=false

python -u src/training/main.py \
    --save-frequency 1     \
    --train-data="data/train_coarse.txt,data/train_fine.txt.00"    \
    --val-data="data/train_fine.txt.01"   \
    --dataset-type="json" \
    --warmup 1000  \
    --batch-size=32  \
    --lr=1e-4  \
    --wd=0.1  \
    --epochs=100   \
    --workers=4   \
    --dp   \
    --multigpu 2,3 \
    --name "demo"