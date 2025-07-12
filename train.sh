#! /bin/bash
DEVICE_ID=$1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python train_sccan.py --config=config/pascal/pascal_split0_resnet50.yaml --viz