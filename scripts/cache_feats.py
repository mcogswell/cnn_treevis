#!/usr/bin/env python
import cPickle as pkl
import pprint
import glob
import os.path as pth

import numpy as np
import matplotlib.pyplot as plt

import skimage.io as io

from jinja2 import Template

import caffe
import caffe.proto.caffe_pb2 as cpb

from recon.reconstruct import FeatBuilder
import recon.util as util
import recon.config

def main():
    '''
    Usage:
        cache_feats.py <net_id> <blob_name> [--topk <K>] [--nfeatures <N>] [--stdout] [--gpu-id <id>]

    Options:
        --topk <K>         How many images to display at once? [default: 9]
        --nfeatures <N>    Only visualize the first N features [default: -1]
        --stdout           Log info to stdout alongside log file [default: false]
        --gpu-id <id>      Id of gpu to use [default: -1]
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    net_id = main_args['<net_id>']
    use_stdout = main_args['--stdout']
    gpu_id = int(main_args['--gpu-id'])
    config = recon.config.config.nets[net_id]
    builder = FeatBuilder(config, gpu_id)

    util.setup_logging('test_{}'.format(net_id), use_stdout=use_stdout)

    blob_name = main_args['<blob_name>']
    topk = int(main_args['--topk'])
    nfeatures = int(main_args['--nfeatures'])
    if nfeatures == -1:
        nfeatures = builder.num_features(blob_name)
    for i in range(nfeatures):
        builder.canonical_image(blob_name, i, topk, 'data/feat/{}_feat{}.jpg'.format(blob_name, i))


if __name__ == '__main__':
    main()
