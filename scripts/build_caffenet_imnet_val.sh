#!/bin/sh
(scripts/build_db.py caffenet_imnet_val conv1 --gpu 0 &)
(scripts/build_db.py caffenet_imnet_val conv2 --gpu 1 &)
(scripts/build_db.py caffenet_imnet_val conv3 --gpu 2 &)
(scripts/build_db.py caffenet_imnet_val conv4 --gpu 3 &)
(scripts/build_db.py caffenet_imnet_val conv5 --gpu 4 &)
