#!/usr/bin/env python

from recon import *
from recon.config import config



def main():
    '''
    Usage:
        api.py maxes <net_id> <blob_name>

    Options:
    '''
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    if main_args['maxes']:
        net_id = main_args['<net_id>']
        blob_name = main_args['<blob_name>']
        rec = Reconstructor(net_id)
        max_idxs = rec.max_idxs(blob_name)
        print('{}:\n{}'.format(blob_name, max_idxs))



if __name__ == '__main__':
    main()
