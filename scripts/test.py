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
        vis.py maxes <net_id> <blob_name> [--topk <K>] [--nfeatures <N>] [--stdout]
        vis.py single <net_id> <blob_names>... [--stdout]
        vis.py graph <net_id> <blob_names>... [--stdout]

    Options:
        --topk <K>         How many images to display at once? [default: 9]
        --nfeatures <N>    Only visualize the first N features [default: -1]
        --stdout    Log info to stdout alongside log file [default: false]
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    net_id = main_args['<net_id>']
    use_stdout = main_args['--stdout']
    rec = Reconstructor(net_id)

    util.setup_logging('test_{}'.format(net_id), use_stdout=use_stdout)

    if main_args['maxes']:
        blob_name = main_args['<blob_name>']
        topk = int(main_args['--topk'])
        nfeatures = int(main_args['--nfeatures'])
        if nfeatures == -1:
            nfeatures = rec.num_features(blob_name)
        for i in range(nfeatures):
            rec.canonical_image(blob_name, i, topk, '/tmp/{}_feat{}.jpg'.format(blob_name, i))
    elif main_args['single']:
        blob_names = main_args['<blob_names>']
        rec.reconstruct(blob_names)
    elif main_args['graph']:
        blob_names = main_args['<blob_names>']
        d = rec.graph()
        keyboard('graph')
        with open('/tmp/graph.json', 'w') as f:
            json.dump(d, f)


if __name__ == '__main__':
    main()
