import numpy as np
import matplotlib.pyplot as plt

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

import config as config
from reconstruct import *

def main():
    reconstruct('specs/train_val.prototxt', 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

if __name__ == '__main__':
    main()
