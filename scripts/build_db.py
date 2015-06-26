import caffe

from recon import Reconstructor

def main():
    '''
    Usage:
        build_db.py <net_id> <blob_names>... [--gpu <id>]

    Options:
        --gpu <id>  The id of the GPU to use [default: -1]
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))
    net_id = main_args['<net_id>']
    blob_names = main_args['<blob_names>']
    gpu_id = int(main_args['--gpu'])

    if gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

    recon = Reconstructor(net_id)
    recon.build_max_act_db(blob_names)

if __name__ == '__main__':
    main()
