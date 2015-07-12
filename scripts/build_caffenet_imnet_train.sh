#!/bin/sh
(scripts/build_db.py caffenet_imnet_train conv1 --gpu 0 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv2 --gpu 1 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv3 --gpu 2 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv4 --gpu 3 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train conv5 --gpu 4 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train fc6 --gpu 2 --topk 16 &)
(scripts/build_db.py caffenet_imnet_train fc7 --gpu 7 --topk 16 &)
