TOTAL_NUM_UPDATES=2000  # Total number of training steps.
WARMUP_UPDATES=250      # Linearly increase LR over this many steps.
LR=2e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=16        # Batch size per GPU.
SEED=1                  # Random seed.
ROBERTA_PATH=roberta/models/roberta.base.orig/model.pt

# we use the --user-dir option to load the task and criterion
# from the examples/roberta/wsc directory:
FAIRSEQ_PATH=../fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/wsc

#CUDA_VISIBLE_DEVICES=0,1,2,3 
fairseq-train data/wino/WSC/ \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --valid-subset val \
    --fp16 --ddp-backend legacy_ddp \
    --user-dir $FAIRSEQ_USER_DIR \
    --task wsc --criterion wsc --wsc-cross-entropy \
    --arch roberta_base --bpe gpt2 --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-update $TOTAL_NUM_UPDATES \
    --log-format simple --log-interval 100 \
    --seed $SEED \
--cpu
