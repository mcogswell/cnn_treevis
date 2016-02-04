#!/bin/sh
if [ "$#" -ge 1 ]; then
    python app.py caffenet_imnet_val --debug --gpu-id $1
else
    python app.py caffenet_imnet_val --debug
fi
