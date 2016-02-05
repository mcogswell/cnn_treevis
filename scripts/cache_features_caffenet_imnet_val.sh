#!/bin/sh
# This can take a long time (a couple hours), but it can go a lot
# faster if you use multiple gpus. To do that you can run
# the `_parallel_1` version of this script and then, once
# all of the jobs it starts finish, the `_parallel_2` version
# of this script. You might want to edit the GPUs each scripts uses.
# NOTE: provide `--gpu -1` to use the CPU
# TODO: combine the two component scripts into one
scripts/build_db.py caffenet_imnet_val conv1 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv2 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv3 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv4 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv5 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc6   --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc7   --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc8   --gpu 0 --topk 16

scripts/cache_feats.py caffenet_imnet_val conv1 --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv2 --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv3 --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv4 --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv5 --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val fc6   --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val fc7   --gpu-id 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val fc8   --gpu-id 0 --topk 16
