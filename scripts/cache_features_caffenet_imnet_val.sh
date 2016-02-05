#!/bin/sh
# This can take a long time (a couple hours), but it can go a lot
# faster if you use multiple gpus. Each of the build_db.py calls
# can be run in parallel, as can the cache_feats.py scripts, but
# don't run cache_feats.py before the corresponding build_db.py
# finishes. Try changing the gpu_ids in this file and puting an
# ampersand at the end of every line, running one block of scripts
# at a time.
# TODO: combine the two component scripts into one
scripts/build_db.py caffenet_imnet_val conv1 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv2 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv3 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv4 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv5 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc6   --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc7   --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc8   --gpu 0 --topk 16

scripts/cache_feats.py caffenet_imnet_val conv1 --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv2 --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv3 --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv4 --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val conv5 --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val fc6   --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val fc7   --gpu 0 --topk 16
scripts/cache_feats.py caffenet_imnet_val fc8   --gpu 0 --topk 16
