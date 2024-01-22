#!/bin/bash

# Define the paths
DATA_DIR="./data"
SAVE_DIR="./cgc_save"
LOG_DIR="./cgc_log"

# Model architecture options: resnet18, resnet50
ARCH="resnet18"
DATASET="cifar100"

# CGC loss weight
LAMBDA=0.5

# Other training parameters
EPOCHS=100
BATCH_SIZE=64
LR=0.03
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
PRINT_FREQ=10
EVALUATE=true
DIST_BACKEND="nccl"
SEED=42

# ???
# --gpu $GPU \

python train_eval_cgc.py \
              -a $ARCH \
              --epochs $EPOCHS \
              -b $BATCH_SIZE \
              --lr $LR \
              --momentum $MOMENTUM \
              --wd $WEIGHT_DECAY \
              -p $PRINT_FREQ \
              --seed $SEED \
              --save_dir $SAVE_DIR \
              --log_dir $LOG_DIR \
              --dataset $DATASET \
              --lambda $LAMBDA \
              $DATA_DIR
