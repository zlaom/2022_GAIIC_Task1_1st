# ------------attr------------ #
GPUS='1'
TOTAL_FOLD=20

for FOLD_ID in 15 16 17 18 19 
do
    echo "fold is: ${FOLD_ID}"
    python project/code/attr_train.py \
        --gpus ${GPUS} \
        --total_fold ${TOTAL_FOLD} \
        --fold_id ${FOLD_ID}
done
