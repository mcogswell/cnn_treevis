#!/bin/sh
(scripts/build_db.py caffenet_imnet_val conv1 --gpu 0 --topk 16 &)
(scripts/build_db.py caffenet_imnet_val conv2 --gpu 1 --topk 16 &)
(scripts/build_db.py caffenet_imnet_val conv3 --gpu 2 --topk 16 &)
(scripts/build_db.py caffenet_imnet_val conv4 --gpu 3 --topk 16 &)
(scripts/build_db.py caffenet_imnet_val conv5 --gpu 4 --topk 16 &)
(scripts/build_db.py caffenet_imnet_val fc6 --gpu 5 --topk 16 &)
(scripts/build_db.py caffenet_imnet_val fc7 --gpu 6 --topk 16 &)
