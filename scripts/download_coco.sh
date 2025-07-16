#!/bin/bash

set -e
cd ..
mkdir -p data
cd data

mkdir -p MSCOCO2014
cd MSCOCO2014

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

mkdir -p annotations
cd annotations

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

mkdir -p train2014 val2014
mv annotations/instances_train2014.json train2014/
mv annotations/instances_val2014.json val2014/