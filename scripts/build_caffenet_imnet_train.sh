#!/bin/sh
(scripts/build_db.py caffenet_imnet_train conv1 --gpu 13 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv2 --gpu 14 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv3 --gpu 15 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv4 --gpu 8 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv5 --gpu 9 --topk 16 &)
