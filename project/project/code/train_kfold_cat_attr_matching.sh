echo Begin Train
python train_kfold_cat_attr_matching.py --gpu 0 --fold_ids 0 1 2 >/dev/null 2>&1 &
python train_kfold_cat_attr_matching.py --gpu 0 --fold_ids 3 4 5 >/dev/null 2>&1 &
python train_kfold_cat_attr_matching.py --gpu 0 --fold_ids 6 7 8 9 >/dev/null 2>&1
echo End Train
# python train_kfold_cat_attr_matching.py --gpu 0 --fold_ids 6 7
# python train_kfold_cat_attr_matching.py --gpu 0 --fold_ids 8 9