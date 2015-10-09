#!/usr/bin/env python
import cPickle as pkl
import glob
import os.path as pth
from pdb import set_trace

import numpy as np

from jinja2 import Template
import flask
from flask import Flask, request, render_template, send_file, jsonify, url_for
app = Flask(__name__)

from urllib import unquote

from recon import *
from recon.config import config



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
    return render_template('vis.html', blob_name=blob_name, act_id=act_id)


@app.route('/vis/tree.json')
def json_tree():
    blob_name = request.args.get('blob_name', '')
    act_id = int(request.args.get('act_id', ''))
    # should input layer_name, not blob_name
    tree = vis_tree.tree(blob_name, act_id)
    return jsonify(tree)


@app.route('/vis/tree/maxes')
def json_tree_maxes():
    blob_name = request.args.get('blob_name', '')
    maxes = vis_tree.max_idxs(blob_name)
    return jsonify(maxes=maxes)





def main():
    '''
    Usage:
        app.py <net_id> [--debug]

    Options:
        --debug     Launch the app in debug mode? [default: false]
    '''
    global rec, vis_tree
    import docopt, textwrap
    main_args = docopt.docopt(textwrap.dedent(main.__doc__))

    rec = Reconstructor(main_args['<net_id>'])
    vis_tree = VisTree(main_args['<net_id>'])
    app.run(host='fukushima.ece.vt.edu', debug=main_args['--debug'])


if __name__ == '__main__':
    main()
