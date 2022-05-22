# ------------order------------ #
PRETRAIN_SEED=0
GPUS='3'
CKPT_FILE='temp/tmp_data/lhq_output/title_pretrain/order/order_seed0_acc0.9318_loss0.26482.pth'

for SEED in 0 1 2 3 4
do
    echo "seed is: ${SEED}"
    python project/code/title_finetune_freeze_nobug_order.py \
        --gpus ${GPUS} \
        --seed ${SEED} \
        --pretrain_seed ${PRETRAIN_SEED} \
        --ckpt_file ${CKPT_FILE}
done


# ------------fold 0------------ #
PRETRAIN_SEED=0
GPUS='3'
FOLD_ID=0
CKPT_FILE='temp/tmp_data/lhq_output/title_pretrain/fold0/fold0_seed0_acc0.9318_loss0.24995.pth'

for SEED in 0 1 2 3 4
do
    echo "seed is: ${SEED}"
    python project/code/title_finetune_freeze_nobug_seed.py \
        --gpus ${GPUS} \
        --seed ${SEED} \
        --fold_id ${FOLD_ID} \
        --pretrain_seed ${PRETRAIN_SEED} \
        --ckpt_file ${CKPT_FILE}
done


# # ------------fold 5------------ #
# PRETRAIN_SEED=0
# GPUS='3'
# FOLD_ID=5
# CKPT_FILE='temp/tmp_data/lhq_output/title_pretrain/fold5/fold5_seed0_acc0.9328_loss0.27301.pth'

# for SEED in 0 1 2 3 4
# do
#     echo "seed is: ${SEED}"
#     python project/code/title_finetune_freeze_nobug_seed.py \
#         --gpus ${GPUS} \
#         --seed ${SEED} \
#         --fold_id ${FOLD_ID} \
#         --pretrain_seed ${PRETRAIN_SEED} \
#         --ckpt_file ${CKPT_FILE}
# done


# # ------------fold 3------------ #
# PRETRAIN_SEED=0
# GPUS='3'
# FOLD_ID=3
# CKPT_FILE='temp/tmp_data/lhq_output/title_pretrain/fold3/fold3_seed0_acc0.9190_loss0.32237.pth'

# for SEED in 0 1 2 3 4
# do
#     echo "seed is: ${SEED}"
#     python project/code/title_finetune_freeze_nobug_seed.py \
#         --gpus ${GPUS} \
#         --seed ${SEED} \
#         --fold_id ${FOLD_ID} \
#         --pretrain_seed ${PRETRAIN_SEED} \
#         --ckpt_file ${CKPT_FILE}
# done