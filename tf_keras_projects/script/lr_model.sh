#!/bin/sh # 获取商品ctr分值 
echo "execute $BASH_SOURCE"
cd "`dirname $0`/.."
pwd

PYBIN="/anaconda3/bin/python"

MODE="TRAIN"
MODE="EVAL"
MODE="PREDICT"
${PYBIN} lr_model.py                    \
    --name "LR"                         \
    --model "ml_common.model.lr_model"  \
    --reader "ml_common.data.lr_reader" \
    --mode "$MODE"                      \
    --input-size "768"                  \
    --batch-size "128"                  \
    --num-epoch "1"                     \
    --log-dir "./log/lr"                \
    --train-data "#"                    \
    --eval-data "#" 
