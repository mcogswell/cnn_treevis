#!/bin/sh
(scripts/build_db.py caffenet_1000 conv1 --gpu 0 &)
(scripts/build_db.py caffenet_1000 conv2 --gpu 1 &)
(scripts/build_db.py caffenet_1000 conv3 --gpu 2 &)
(scripts/build_db.py caffenet_1000 conv4 --gpu 3 &)
(scripts/build_db.py caffenet_1000 conv5 --gpu 4 &)
