#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Error: need 3 parameters"
    echo "Usage: $0 <device_id> <split> <shot>"
    echo "device_id: GPU ID"
    echo "split: 0-3, split of dataset"
    echo "shot: [1|5], support image num"
    echo "Example: $0 0 0 1"
    exit 1
fi

DEVICE_ID=$1
SPLIT=$2
SHOT=$3

if ! [[ "$DEVICE_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: device_id must be a number"
    exit 1
fi

if ! [[ "$SPLIT" =~ ^[0-3]$ ]]; then
    echo "Error: split must be 0-3"
    exit 1
fi

if ! [[ "$SHOT" =~ ^[1|5]$ ]]; then
    echo "Error: shot must be 1 or 5"
    exit 1
fi

mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_split${SPLIT}_${SHOT}shot_${TIMESTAMP}.log"

if [ "$SHOT" -eq 5 ]; then
    SHOT_STR="_5s"
else
    SHOT_STR=""
fi

CUDA_VISIBLE_DEVICES=$DEVICE_ID python train_sccan.py \
  --config=config/pascal/pascal_split${SPLIT}_resnet50${SHOT_STR}.yaml \
  --viz 2>&1 | tee "$LOG_FILE"
