#!/bin/bash
DEVICE_ID=$1
SPLIT=$2

mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_split_${SPLIT}_${TIMESTAMP}.log"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python train_sccan.py \
  --config=config/pascal/pascal_split${SPLIT}_resnet50.yaml \
  --viz 2>&1 | tee "$LOG_FILE"
