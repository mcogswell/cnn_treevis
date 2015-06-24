import numpy as np
import matplotlib.pyplot as plt

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

import config as config
from reconstruct import *

def main():
    '''
    Usage:
        vis.py <layer_name>
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    layer_name = main_args['<layer_name>']

    reconstruct('specs/train_val.prototxt',
                'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                layer_name,
                'results/',
                # NORMAL, DECONV, GUIDED
                relu_type='GUIDED')

if __name__ == '__main__':
    main()
