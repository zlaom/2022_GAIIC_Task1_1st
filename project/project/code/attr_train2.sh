# ------------attr------------ #
GPUS='1'
TOTAL_FOLD=20

for FOLD_ID in 10 11 12 13 14 
do
    echo "fold is: ${FOLD_ID}"
    python project/code/attr_train.py \
        --gpus ${GPUS} \
        --total_fold ${TOTAL_FOLD} \
        --fold_id ${FOLD_ID}
done
