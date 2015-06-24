import sys
sys.path.append('lib/')

import numpy as np
import matplotlib.pyplot as plt

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

def load_mean_image():
    bp = cpb.BlobProto()
    with open('caffe/data/ilsvrc12/imagenet_mean.binaryproto', 'r') as f:
        bp.ParseFromString(f.read())
    return np.array(bp.data).reshape([3, 256, 256]).transpose([1, 2, 0])

