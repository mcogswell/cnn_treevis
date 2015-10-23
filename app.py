#!/usr/bin/env python
import cPickle as pkl
import glob
import os.path as pth
from pdb import set_trace
import cStringIO as StringIO
import io
import socket

import numpy as np
import scipy

from jinja2 import Template
import flask
from flask import Flask, request, render_template, send_file, jsonify, url_for
app = Flask(__name__)

from urllib import unquote

from recon import *
from recon.config import config



@app.route('/imgs/feat/<path:path>')
def img_feat(path):
    return send_file(pth.join('data/feat/', path))


@app.route('/imgs/gallery/<path:path>')
def img_gallery(path):
    return send_file(pth.join('data/gallery/', path))


@app.route('/gallery')
def gallery():
    img_fnames = [pth.basename(fname) for fname in glob.glob('data/gallery/*')]
    # TODO: better/parameterized defaults
    return render_template('gallery.html', img_fnames=img_fnames, imgs_per_row=5, blob_name='conv5', act_id=4)


@app.route('/vis')
def vis_activation():
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))
    return render_template('vis.html', blob_name=blob_name, act_id=act_id)


@app.route('/vis/tree/get')
def json_tree():
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))
    # should input layer_name, not blob_name
    tree = get_vis_tree(net_id).tree(blob_name, act_id)
    return jsonify(tree)


@app.route('/vis/tree/maxes')
def json_tree_maxes():
    blob_name = request.args.get('blob_name', '')
    maxes = get_vis_tree(net_id).max_idxs(blob_name)
    return jsonify(maxes=maxes)


@app.route('/vis/tree/expand')
def json_tree_expand():
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))
    # should input layer_name, not blob_name
    children = get_vis_tree(net_id).expand(blob_name, act_id)
    return jsonify(children=children)


@app.route('/vis/tree/reconstruction')
def json_tree_reconstruction():
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))
    layer_name = config['nets'][net_id]['blob_name_to_layer_name'][blob_name]
    recons = get_vis_tree(net_id).reconstruction(layer_name, blob_name, act_id)

    f = io.BytesIO()
    scipy.misc.imsave(f, recons['reconstruction'], format='jpeg')
    f.seek(0)
    return send_file(f,
                     attachment_filename='recon_{}_{}.jpg'.format(blob_name, act_id),
                     mimetype='image/jpeg')


_vis_trees = {}
def get_vis_tree(img_fname):
    if img_fname in _vis_trees:
        return _vis_trees[img_fname]
    else:
        vis_tree = VisTree(net_id)
        _vis_trees[img_fname] = vis_tree
        return vis_tree



def main():
    '''
    Usage:
        app.py <net_id> [--debug]

    Options:
        --debug     Launch the app in debug mode? [default: false]
    '''
    global net_id
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))
    net_id = main_args['<net_id>']

    hostname = socket.gethostname()
    app.run(host=hostname, debug=main_args['--debug'])


if __name__ == '__main__':
    main()
