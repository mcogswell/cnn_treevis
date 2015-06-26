import numpy as np

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard


def load_mean_image(fname):
    bp = cpb.BlobProto()
    with open(fname, 'r') as f:
        bp.ParseFromString(f.read())
    # TODO: don't always assume BGR
    mean = np.array(bp.data).reshape([3, 256, 256]).transpose([1, 2, 0])
    return mean[15:-14, 15:-14]

