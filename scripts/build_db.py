#!/usr/bin/env python
import caffe

from recon import Reconstructor
from recon import util

def main():
    '''
    Usage:
        build_db.py <net_id> <blob_names>... [--gpu <id>] [--topk <k>]

    Options:
        --gpu <id>  The id of the GPU to use [default: -1]
        --topk <k>  The number of top image to store [default: 5]
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))
    net_id = main_args['<net_id>']
    blob_names = main_args['<blob_names>']
    gpu_id = int(main_args['--gpu'])
    top_k = int(main_args['--topk'])

    if gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

    util.setup_logging('{}_{}'.format(net_id, '_'.join(blob_names)))

    recon = Reconstructor(net_id)
    recon.build_max_act_db(blob_names, k=top_k)

if __name__ == '__main__':
    main()
