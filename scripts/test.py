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

from recon import *
from recon.config import config

def main():
    '''
    Usage:
        vis.py maxes <net_id> <blob_name> [--topk <K>] [--nfeatures <N>] [--stdout]
        vis.py single <net_id> <blob_names>... [--stdout]
        vis.py graph <net_id> <blob_names>... [--stdout]
        vis.py bigraph <net_id> <blob_name> <act_id> [--stdout]

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
        with open('/tmp/graph.json', 'w') as f:
            json.dump(d, f)
    elif main_args['bigraph']:
        blob_name = main_args['<blob_name>']
        act_id = int(main_args['<act_id>'])
        blob_to_idx = {'conv{}'.format(i): i-1 for i in [1, 2, 3, 4, 5]}
        if blob_name not in blob_to_idx.keys():
            raise Exception('only blobs conv1 - conv5 supported')
        with open('data/parvis_edges.pkl', 'r') as f:
            edge_data = pkl.load(f)
        adj_mats = edge_data['adj_mats']
        mat = adj_mats[blob_to_idx[blob_name]]
        print('top {} features:'.format(blob_name))
        print(mat.sum(axis=1).argsort()[::-1][:40])
        # note that this doesn't take extremely low activations into account
        prev_top_act_values = np.sort(mat[act_id])[::-1]
        prev_top_act_idxs = np.argsort(mat[act_id])[::-1]

        prev_layer_map = {
            # TODO: make these work
            #'fc7': 'fc6',
            #'fc6': 'conv5',
            'conv5': 'conv4',
            'conv4': 'conv3',
            'conv3': 'conv2',
            'conv2': 'conv1',
            'conv1': 'data',
        }

        def find_recon(bname, fidx):
            imgs = glob.glob('data/recon/recon_{}_ord*_feat{}.jpg'.format(bname, fidx))
            assert len(imgs) <= 1
            if imgs:
                return pth.basename(imgs[0])
            else:
                return ''

        root = {
            'imname': '{}_feat{}.jpg'.format(blob_name, act_id),
            'caption': 'blob: {}, feat: {}'.format(blob_name, act_id),
            'recon_imname': find_recon(blob_name, act_id),
        }

        prev_name = prev_layer_map[blob_name]
        prevs = []
        for idx, act in zip(prev_top_act_idxs[:5], prev_top_act_values):
            prevs.append({
                'imname': '{}_feat{}.jpg'.format(prev_name, idx),
                'caption': 'blob: {}, feat: {}, act: {}'.format(prev_name, idx, act),
                'recon_imname': find_recon(prev_name, idx),
            })
        vis_info = {
            'root': root,
            'prevs': prevs,
        }

        with open('web/index.html', 'r') as f:
            spec = Template(f.read())
        with open('web/public/{}_{}.html'.format(blob_name, act_id), 'w') as f:
            f.write(spec.render(vis_info))


if __name__ == '__main__':
    main()
