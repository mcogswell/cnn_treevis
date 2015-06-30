#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

import recon.config as config
from recon import *

def main():
    '''
    Usage:
        vis.py <net_id> <blob_name>
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    net_id = main_args['<net_id>']
    blob_name = main_args['<blob_name>']

    rec = Reconstructor(net_id)
    for i in range(30):
        rec.canonical_image(blob_name, i, 5, '/tmp/test_{}.jpg'.format(i))


if __name__ == '__main__':
    main()
