#!/bin/bash

set -e
cd ..
mkdir -p data
cd data

if [ ! -f VOCtrainval_11-May-2012.tar ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
fi

tar -xf VOCtrainval_11-May-2012.tar

mkdir -p VOCdevkit2012/VOC2012

mv VOCdevkit/VOC2012/* VOCdevkit2012/VOC2012/

rm -rf VOCdevkit