#!/usr/bin/env python
import cPickle as pkl
import glob
import os.path as pth
from pdb import set_trace
import io
import socket

import numpy as np
import scipy

from jinja2 import Template
import flask
from flask import Flask, request, render_template, send_file, jsonify, url_for, send_from_directory
app = Flask(__name__)

from urllib import unquote

from recon import *
from recon.reconstruct import get_path, get_path_id
from recon.config import config


######################################
# Retrieve cached images and static files

@app.route('/imgs/feat/<path:path>')
def img_feat(path):
    '''
    Return an image with a grid of deconv vis's and corresponding patches
    which are typical patches for this feature

    # Args
        path: Name of the feature file for the visualized neuron
    '''
    return send_file(pth.join('data/feat/', path))

@app.route('/imgs/gallery/<path:path>')
def img_gallery(path):
    '''
    Serve images from the gallery (in the `data/gallery/` directory)

    # Args
        path: image name relative to gallery directory
    '''
    return send_file(pth.join('data/gallery/', path))

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


######################################
# Serve complete html pages

@app.route('/gallery')
def gallery():
    '''
    Show all available images
    '''
    img_fnames = [pth.basename(fname) for fname in glob.glob('data/gallery/*')]
    return render_template('gallery.html', img_fnames=img_fnames, imgs_per_row=5)

@app.route('/vis/<path:img_id>/overview')
def vis_overview(img_id):
    '''
    Overview visualization of top activations for each layer w.r.t. a particular image

    # Args
        img_id: Name of image
    '''
    num_maxes = int(request.args.get('num_maxes', '5'))
    tree = get_vis_tree(net_id, img_id)
    layers = [dict(layer) for layer in tree.config['layers'].itervalues() if layer['include_in_overview']]
    layers.sort(key=lambda l: -l['idx'])
    for layer in layers:
        blob_name = layer['blob_name']
        max_act_ids = get_vis_tree(net_id, img_id).max_blob_idxs(blob_name)[:num_maxes]
        max_acts = []
        for act_id in max_act_ids:
            path = [(blob_name, act_id)]
            max_acts.append({'act_id': act_id, 'path_id': get_path_id(path)})
        layer['max_acts'] = max_acts
    return render_template('overview.html', layers=layers, imgs_per_row=5, img_id=img_id)

@app.route('/vis/<path:img_id>')
def vis(img_id):
    '''
    Detailed vis page for a particular net, image, and neuron

    # Args
        img_id: Name of image
    '''
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))
    root_path = [(blob_name, act_id)]
    root_id = get_path_id(root_path)
    return render_template('vis.html',
                           root_path_id=root_id,
                           img_id=img_id,
                           root_blob_name=blob_name,
                           root_act_id=act_id)


######################################
# Retrieve/compute misc visualization details

@app.route('/vis/<path:img_id>/img.jpg')
def vis_img(img_id):
    '''
    Serve the visualized image (resized for network).

    # Args
        img_id: Name of image
    '''
    img = get_vis_tree(net_id, img_id).image()
    return send_img(img, 'img.jpg')

@app.route('/vis/<path:img_id>/labels.json')
def vis_labels(img_id):
    '''
    Get the CNN's top labels for this image.

    # Args
        img_id: Name of image
    '''
    labels = get_vis_tree(net_id, img_id).labels()
    label_str = ''.join(['<p>' + label + '</p>' for label in labels])
    return jsonify(labels=label_str)


######################################
# Interact with the tree of gradient images. See `VisTree`

@app.route('/vis/<path:img_id>/tree/children')
def json_tree_children(img_id):
    '''
    Retrieve the info of a node's children.

    # Args
        img_id: Name of image
    '''
    path_id = request.args.get('path_id', None)
    path = get_path(path_id)
    tree = get_vis_tree(net_id, img_id)
    children = tree.children_from_path(path)
    for child in children:
        child['path_id'] = get_path_id(child['path'])
    return jsonify({'children': children})

@app.route('/vis/<path:img_id>/tree/reconstruction')
def json_tree_reconstruction(img_id):
    '''
    Retrieve the reconstruction for a particular path

    # Args
        img_id: Name of image
    '''
    path_id = request.args.get('path_id', None)
    path = get_path(path_id)
    recons = get_vis_tree(net_id, img_id).reconstruction(path)
    return send_img(recons, 'recon_{}.jpg'.format(path_id))


######################################
# Helpers

def send_img(img, fname):
    '''
    Serve a numpy array as a jpeg image

    # Args
        img: Image (numpy array)
        fname: Name of the sent file
    '''
    f = io.BytesIO()
    scipy.misc.imsave(f, img, format='jpeg')
    f.seek(0)
    return send_file(f, attachment_filename=fname,
                     mimetype='image/jpeg')

_vis_trees = {}
def get_vis_tree(net_id, img_id):
    '''
    Retrieve the tree of reconstructions for the net and image

    # Args
        net_id: Net name (see lib/recon/config.py)
        img_id: Image name
    '''
    key = '{}_{}'.format(net_id, img_id)
    if key in _vis_trees:
        return _vis_trees[key]
    else:
        img_fname = pth.join('data/gallery/', img_id)
        vis_tree = VisTree(net_id, img_fname, gpu_id, mult_mode)
        _vis_trees[key] = vis_tree
        return vis_tree



def main():
    '''
    Usage:
        app.py <net_id> [--debug] [--gpu-id=<id>] [--mult-mode=<meth>]

    Options:
        --debug                 Launch the app in debug mode? [default: false]
        --gpu-id=<id>           Id of GPU to use or -1 for CPU [default: -1]
        --mult-mode=<meth>    Multiplier mode (See VisTree()) [default: auto_1]
    '''
    global net_id, gpu_id, mult_mode
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))
    net_id = main_args['<net_id>']
    gpu_id = int(main_args['--gpu-id'])
    mult_mode = main_args['--mult-mode']

    hostname = socket.gethostname()
    app.run(host=hostname, debug=main_args['--debug'])


if __name__ == '__main__':
    main()
