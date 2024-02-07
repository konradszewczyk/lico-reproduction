#!/bin/bash

python -m eval.consistency_evaluation \
    --ckpt-path checkpoint/imagenets50-ablation/baseline-imgnets50.ckpt \
    --dataset imagenet-s50 \
    --save-dir r2baseline-imgnets50

python -m eval.consistency_evaluation \
    --ckpt-path checkpoint/imagenets50-ablation/lico-imgnets50.ckpt \
    --dataset imagenet-s50 \
    --save-dir r2lico-imgnets50

python -m eval.consistency_evaluation \
    --ckpt-path checkpoint/imagenets50-ablation/no-mm-imgnets50.ckpt \
    --dataset imagenet-s50 \
    --save-dir r2no-mm-imgnets50

python -m eval.consistency_evaluation \
    --ckpt-path checkpoint/imagenets50-ablation/no-ot-imgnets50.ckpt \
    --dataset imagenet-s50 \
    --save-dir r2no-ot-imgnets50

python -m eval.consistency_evaluation \
    --ckpt-path checkpoint/baseline-seed-1.ckpt \
    --dataset imagenet-s50 \
    --save-dir baseline-seed-1

python -m eval.consistency_evaluation \
    --ckpt-path checkpoint/baseline-seed-3.ckpt \
    --dataset imagenet-s50 \
    --save-dir baseline-seed-3
