# ------------attr------------ #
GPUS='0'
TOTAL_FOLD=20

for FOLD_ID in 5 6 7 8 9 
do
    echo "fold is: ${FOLD_ID}"
    python project/code/attr_train.py \
        --gpus ${GPUS} \
        --total_fold ${TOTAL_FOLD} \
        --fold_id ${FOLD_ID}
done
