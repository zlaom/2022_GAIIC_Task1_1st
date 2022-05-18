# PRETRAIN_SEED=1
# GPUS='3'
# FOLD_ID=5
# CKPT_FILE='output/pretrain/title/2tasks_seed/fold5/fold5_seed1_acc0.9328_loss0.28734.pth'

# for SEED in 0 1 2 3 4
# do
#     echo "seed is: ${SEED}"
#     python title_finetune_2tasks.py \
#         --gpus ${GPUS} \
#         --seed ${SEED} \
#         --fold_id ${FOLD_ID} \
#         --pretrain_seed ${PRETRAIN_SEED} \
#         --ckpt_file ${CKPT_FILE}
# done


# order
PRETRAIN_SEED=1
GPUS='3'
CKPT_FILE='output/pretrain/title/2tasks_seed/order/order_seed1_acc0.9290_loss0.27155.pth'

for SEED in 0 1 2 3 4
do
    echo "seed is: ${SEED}"
    python title_finetune_2tasks_order.py \
        --gpus ${GPUS} \
        --seed ${SEED} \
        --pretrain_seed ${PRETRAIN_SEED} \
        --ckpt_file ${CKPT_FILE}
done