#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import skimage.io as io

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

from recon import *
from recon.config import config

def main():
    '''
    Usage:
        vis.py (maxes|single) <net_id> <blob_name> [--topk <K>] [--nfeatures <N>]

    Options:
        --topk <K>         How many images to display at once? [default: 9]
        --nfeatures <N>    Only visualize the first N features [default: 30]
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    net_id = main_args['<net_id>']
    blob_name = main_args['<blob_name>']
    topk = int(main_args['--topk'])
    nfeatures = int(main_args['--nfeatures'])

    rec = Reconstructor(net_id)
    if main_args['maxes']:
        for i in range(nfeatures):
            rec.canonical_image(blob_name, i, topk, '/tmp/test_{}.jpg'.format(i))
    elif main_args['single']:
        imgs, recons = rec.reconstruct(blob_name)
        for i, recon in enumerate(recons):
            mult = config.nets[net_id].blob_multipliers[blob_name]
            io.imsave('/tmp/recon_{}.jpg'.format(i), rec._showable(mult*recon))


if __name__ == '__main__':
    main()
