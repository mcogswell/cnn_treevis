#!/usr/bin/env python
import cPickle as pkl
import glob
import os.path as pth
from pdb import set_trace

import numpy as np

from jinja2 import Template
from flask import Flask, request, render_template, send_file
app = Flask(__name__)

from recon import *
from recon.config import config



@app.route('/maxes/')
@app.route('/maxes/<blob_name>')
def maxes(blob_name='conv5'):
    max_idxs = rec.max_idxs(blob_name)
    return '{}:\n{}'.format(blob_name, max_idxs)


@app.route('/imgs/feat/<path:path>')
def img_feat(path):
    return send_file(pth.join('data/feat/{}'.format(path)))

'''
@app.route('/imgs/recon/<path:path>')
def img_recon(path):
    # TODO
    return send_file(pth.join('data/recon/{}'.format(path)))
'''


@app.route('/vis')
def vis_activation():
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))

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

    return render_template('index.html', **vis_info)


def main():
    '''
    Usage:
        app.py <net_id>

    Options:
    '''
    global rec
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    rec = Reconstructor(main_args['<net_id>'])
    app.run(host='fukushima.ece.vt.edu', debug=True)


if __name__ == '__main__':
    main()
