#!/usr/bin/env python
import caffe

from recon.reconstruct import build_max_act_db
import recon.config
from recon import util

def main():
    '''
    Usage:
        build_db.py <net_id> <blob_name> [--gpu <id>] [--topk <k>]
            [--stdout]

    Options:
        --gpu <id>  The id of the GPU to use [default: -1]
        --topk <k>  The number of top image to store [default: 5]
        --stdout    Log info to stdout alongside log file [default: false]
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))
    net_id = main_args['<net_id>']
    blob_name = main_args['<blob_name>']
    gpu_id = int(main_args['--gpu'])
    top_k = int(main_args['--topk'])
    use_stdout = main_args['--stdout']

    if gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

    util.setup_logging('{}_{}'.format(net_id, blob_name), use_stdout=use_stdout)

    config = recon.config.config.nets[net_id]
    build_max_act_db(blob_name, config, k=top_k)

if __name__ == '__main__':
    main()
