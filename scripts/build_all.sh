#!/bin/sh
if [[ $# -ne 1 ]]; then
    echo "Usage: build_all.sh <net_name>"
    exit 1
fi
NET=$1

(scripts/build_db.py "$NET" conv1 --gpu 0 --topk 16 &)
(scripts/build_db.py "$NET" conv2 --gpu 1 --topk 16 &)
(scripts/build_db.py "$NET" conv3 --gpu 2 --topk 16 &)
(scripts/build_db.py "$NET" conv4 --gpu 3 --topk 16 &)
(scripts/build_db.py "$NET" conv5 --gpu 4 --topk 16 &)
(scripts/build_db.py "$NET" fc6 --gpu 5 --topk 16 &)
(scripts/build_db.py "$NET" fc7 --gpu 6 --topk 16 &)
