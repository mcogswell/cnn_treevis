#!/bin/sh
scripts/cache_feats.py caffenet_imnet_val conv1 --gpu-id 0 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val conv2 --gpu-id 1 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val conv3 --gpu-id 2 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val conv4 --gpu-id 3 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val conv5 --gpu-id 4 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val fc6   --gpu-id 5 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val fc7   --gpu-id 6 --topk 16 &
scripts/cache_feats.py caffenet_imnet_val fc8   --gpu-id 7 --topk 16 &
