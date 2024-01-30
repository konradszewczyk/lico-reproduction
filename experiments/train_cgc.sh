#!/bin/bash

# Define the paths
DATA_DIR="C:/Users/Mikhail/Datasets/ImagenetS/ImageNetS50"
#DATA_DIR="./data"
SAVE_DIR="./cgc_save"
LOG_DIR="./cgc_log"

# Model architecture options: resnet18, resnet50
ARCH="resnet50"
DATASET="imagenet-s50"

# CGC loss weight
LAMBDA=0.5

# Other training parameters
EPOCHS=3
BATCH_SIZE=128
LR=0.03
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
PRINT_FREQ=10
DIST_BACKEND="nccl"
SEED=42

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
