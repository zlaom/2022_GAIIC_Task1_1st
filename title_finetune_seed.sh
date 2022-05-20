# seed
PRETRAIN_SEED=0
GPUS='3'
FOLD_ID=5
CKPT_FILE='output/pretrain/title/2tasks_day17_1rep_2rep_2wordloss/fold5_seed0/0l6lexp6_acc_0.9328.pth'

for SEED in 0 1 2 3 4
do
    echo "seed is: ${SEED}"
    python title_finetune_2tasks.py \
        --gpus ${GPUS} \
        --seed ${SEED} \
        --fold_id ${FOLD_ID} \
        --pretrain_seed ${PRETRAIN_SEED} \
        --ckpt_file ${CKPT_FILE}
done


# order
# PRETRAIN_SEED=0
# GPUS='3'
# CKPT_FILE='output/pretrain/title/2tasks_nobug/1rep_2rep_2wordloss/0l6lexp6_acc_0.9318.pth'

# for SEED in 0 1 2 3 4 11 43
# do
#     echo "seed is: ${SEED}"
#     python title_finetune_2tasks_order.py \
#         --gpus ${GPUS} \
#         --seed ${SEED} \
#         --pretrain_seed ${PRETRAIN_SEED} \
#         --ckpt_file ${CKPT_FILE}
# done