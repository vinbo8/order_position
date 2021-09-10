#!/bin/bash
#SBATCH --account=project_2000582
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

set -o errexit
source ~/.bashrc
conda activate ${SCRATCH}/envs/debug

ROOT="${SCRATCH}/order_position/roberta"
ROBERTA_PATH="${ROOT}/models/$1/model.pt"

CUDA_VISIBLE_DEVICES=0 fairseq-train ${ROOT}/glue/unshuffled/$3-bin/ \
    --restore-file ${ROBERTA_PATH} \
    --no-save \
    --max-positions 512 \
    --batch-size $6 \
    --max-tokens 4400 \
    --task sentence_prediction \
    --scramble-position --scramble-partition $2 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $4 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $5 --total-num-update $7 --warmup-updates $8 \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --seed $9
