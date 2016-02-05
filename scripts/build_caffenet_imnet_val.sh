#!/bin/sh
scripts/build_db.py caffenet_imnet_val conv1 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv2 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv3 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv4 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val conv5 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc6 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc7 --gpu 0 --topk 16
scripts/build_db.py caffenet_imnet_val fc8 --gpu 0 --topk 16
