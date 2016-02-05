#!/bin/sh
# replace this with your own lmdb (as in caffe/examples/imagenet/create_imagenet.sh)
IMNET_VAL_LMDB="/srv/share/data/ImageNet/ilsvrc12/ilsvrc12_val_lmdb"
mkdir data/
mkdir data/feat/
cp -r eg_gallery/ data/gallery/
mkdir data/models/
mkdir logs/
ln -s $IMNET_VAL_LMDB data/

cd caffe/
./scripts/download_model_binary.py models/bvlc_reference_caffenet/
./data/ilsvrc12/get_ilsvrc_aux.sh
cd ../

./scripts/cache_features_caffenet_imnet_val.sh
