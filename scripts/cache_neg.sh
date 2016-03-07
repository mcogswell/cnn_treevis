#!/bin/sh
scripts/build_db.py caffenet_imnet_val_neg conv1 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val_neg conv2 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val_neg conv3 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val_neg conv4 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val_neg conv5 --gpu 6 --topk 16
scripts/build_db.py caffenet_imnet_val_neg fc6   --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val_neg fc7   --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val_neg fc8   --gpu 0 --topk 16

scripts/cache_feats.py caffenet_imnet_val conv1 --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val conv2 --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val conv3 --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val conv4 --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val conv5 --gpu-id 6 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val fc6   --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val fc7   --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
scripts/cache_feats.py caffenet_imnet_val fc8   --gpu-id 0 --topk 16 --feat-dir data/neg_feat/
