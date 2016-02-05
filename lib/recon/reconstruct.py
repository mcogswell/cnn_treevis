import time
import os
import os.path as pth
import tempfile
import cPickle as pkl
import math
from bisect import bisect
from collections import defaultdict
from itertools import izip_longest

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.util.montage as montage
import skimage.transform as trans
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity

import networkx as nx
from networkx.readwrite import json_graph

import json

import lmdb

import caffe
import caffe.proto.caffe_pb2 as cpb
import google.protobuf.text_format as text_format

from pdb import set_trace

import recon.config
from recon.config import config, relu_backward_types
from recon.util import load_mean_image, load_ilsvrc12_labels

import logging
logger = logging.getLogger(config.logger.name)


# Keep track of paths through the reconstruction tree with one
# string identifier. This makes it easier to handle paths as identifiers
# in javascript. I can pass around one string instead of a JSON list.
# Especially important for <img src="GET_url?path_id=">
# NOTE: this assumes a bijection between path_ids and paths
_paths_by_id = {}
def _check_path(path):
    for node in path:
        for part in map(str, node):
            if '-' in part or '_' in part:
                raise Exception('Invalid node... can not contain "-" or "_"')

def get_path_id(path):
    _check_path(path)
    path_id = '-'.join(['_'.join(map(str, node)) for node in path])
    _paths_by_id[path_id] = path
    return path_id

def get_path(path_id):
    if path_id in _paths_by_id:
        return _paths_by_id[path_id]
    path = [node.split('_') for node in path_id.split('-')]
    _check_path(path)
    _paths_by_id[path_id] = path
    return path

def _convert_relus(in_param, relu_type=relu_backward_types.GUIDED):
    '''
    Return a new NetParameter with ReLUs suited for visualization
    and a setup to always force backprop.
    '''
    net_param = cpb.NetParameter()
    net_param.CopyFrom(in_param)
    if len(net_param.layer) == 0:
        raise Exception('Network must have at least one layer. Note that ' \
                        'old net specs (with "layers" instead of "layer") ' \
                        'are not yet supported')

    # set relu backprop type and generate temp net
    for layer in net_param.layer:
        if layer.type == 'ReLU':
            layer.relu_param.backprop_type = relu_type

    # otherwise, backprop might not reach the image blob
    net_param.force_backward = True

    return net_param


class VisTree(object):
    '''
    Keep track of explored reconstructions and allow efficient exploration
    of the space by reducing duplication of caffe computation.

    NOTE: ZF means Zeiler/Fergus in reference to "Visualizing and Understanding Convolutional Networks."

    Each node of the tree corresponds to a generalized ZF style visualization.
    A root is exactly a ZF style vis, but it's children/descendants aren't quite.
    Instead of masking all but one pixel in a feature map, a descendant masks all
    gradient images between it and the root in a way that emphasizes features
    connected to the root. For convenience, one of these nodes is indicated by the
    full `path` from root to descendant.

    NOTE: Operations do not modify forward data, so only calls to backward need be made.
    '''

    def __init__(self, net_id, img_fname, gpu_id, multiplier_mode='auto_1'):
        '''
        # Args
            net_id: Network to inspect (lib/recon/config.py)
            img_fname: Image to inspect (data/gallery/)
            gpu_id: Id of GPU to use
            multiplier_mode: see self._filter_feature()
        '''
        self.net_id = net_id
        self.gpu_id = gpu_id
        self.multiplier_mode = multiplier_mode
        self.config = config.nets[self.net_id]
        self.net_param = self._load_param(with_data=False)
        self.mean = load_mean_image(self.config.mean_fname)
        self._labels = load_ilsvrc12_labels(self.config.labels_fname)
        self._set_image(img_fname)
        self.net.forward()
        self.dag = nx.DiGraph()
        self._reconstructions = {}

    #############################
    # Public api

    def labels(self, top_k=5):
        '''
        Return a list of the top k labels assigned to the image

        # Args
            top_k: Number of labels to return, with most likely first
        '''
        prob = self.net.blobs[self.config['prob_blob_name']].data[0].flatten()
        top_idxs = prob.argsort()[::-1][:top_k]
        template = '{} ({:.2f}, {})'
        return [template.format(self._labels[i], prob[i], i) for i in top_idxs]

    def image(self):
        '''
        Return the image visualized by this net
        '''
        img_blob = self.net.blobs[self.config['image_blob_name']]
        return self._showable(img_blob.data[0])

    def reconstruction(self, path):
        '''
        Visualize the given blob/feature pairs in the deconv fashion with layers
        specified in `path` masked to specific features.

        ZF vis is a special case where path consists of one node: [(blob, feature id)]

        Return a reconstruction image (numpy array)
        '''
        self._backprop_path(path)

        # finish backprop to image layer
        top_node = path[-1]
        top_id = self._node_to_layer_id(top_node)
        top_layer_name = self.config['layers'][top_id]['layer_name']
        self.net.backward(start=top_layer_name)

        # cache the visualization
        img_blob = self.net.blobs[self.config['image_blob_name']]
        reconstruction = np.copy(img_blob.diff[0, :, :, :])
        reconstruction = self._showable(reconstruction)
        path_id = get_path_id(path)
        self._reconstructions[path_id] = {
            'reconstruction': reconstruction,
            #'bbox': bbox,
        }
        return reconstruction

    def max_blob_idxs(self, blob_name):
        '''
        Return a list of feature indices which maximally activate the blob.
        '''
        blob = self.net.blobs[blob_name]
        if len(blob.data.shape) == 2:
            features = blob.data
        elif len(blob.data.shape) == 4:
            # NOTE: might want to do this in different ways
            features = blob.data.max(axis=(2, 3))
        return list(features[0].argsort()[::-1])

    def children_from_path(self, path, num_children=5):
        layers = self.config['layers']
        top_id = self._node_to_layer_id(path[-1])
        bottom_id = layers[top_id]['prev_layer_id']
        weights = self._weight_child_neurons(path, bottom_id, num_children)
        sorted_weights = weights.argsort()[::-1]
        children = []
        for child_i in range(num_children):
            child_path = path + [(bottom_id, sorted_weights[child_i])]
            children.append({
                'path': child_path,
                'blob_name': layers[bottom_id]['blob_name'],
                'act_id': sorted_weights[child_i],
            })
        return children

    #############################
    # Helpers

    def _node_to_layer_id(self, node):
        return node[0]

    def _node_to_act_id(self, node):
        return int(node[1])

    # deconv vis stuff

    def _backprop_path(self, path):
        example_i = 0

        # backprop, re-focusing on particular features at each step
        for path_i in range(len(path)):
            # figure out what to backprop and what to filter
            layers = self.config['layers']
            top_node = path[path_i]
            top_id = self._node_to_layer_id(top_node)
            top_blob_name = layers[top_id]['blob_name']
            top_layer_name = layers[top_id]['layer_name']
            top_act_id = self._node_to_act_id(top_node)
            is_last_node = (path_i + 1 == len(path))
            # run filtering and backprop
            # NOTE: Batches might have different layers to start from,
            # so this check will be insufficient when that feature is implemented.
            if path_i == 0:
                self._set_max_pixel(example_i, top_act_id, top_blob_name)
            else:
                self._filter_feature(example_i, top_act_id, top_blob_name, multiplier_mode=self.multiplier_mode)
            if not is_last_node:
                bottom_node = path[path_i + 1]
                bottom_id = self._node_to_layer_id(bottom_node)
                bottom_layer_name = layers[bottom_id]['layer_name']
                self.net.backward(start=top_layer_name, end=bottom_layer_name)

    def _filter_feature(self, example_i, feature_idx, blob_name, multiplier_mode):
        '''
        Mask a diff

        # Args
            example_i: Index of example to mask
            feature_idx: Index of feature to keep
            blob_name: Name of blob to mask
            multiplier_mode: How should the gradient be scaled?
                auto_1: Set the magnitude of single feature gradient equal
                    to the magnitude of all gradients before masking.
                man_1: Look up the gradient magnitude in self.config.
        '''
        blob = self.net.blobs[blob_name]
        if multiplier_mode == 'auto_1':
            total = abs(blob.diff[example_i]).sum()
            total_idx = abs(blob.diff[example_i, feature_idx]).sum()
            mult = total / total_idx
            assert mult >= 1.0
        elif multiplier_mode == 'man_1':
            mult = self.config.blob_multipliers[blob_name]
        else:
            raise Exception('unknown multiplier_mode {}'.format(multiplier_mode))
        blob.diff[example_i, :feature_idx] = 0
        blob.diff[example_i, feature_idx+1:] = 0
        blob.diff[example_i] *= mult

    def _set_max_pixel(self, example_i, feature_idx, blob_name):
        blob = self.net.blobs[blob_name]
        blob.diff[example_i] = 0
        mult = self.config.blob_multipliers[blob_name]
        if len(blob.data.shape) == 2:
            blob.diff[example_i, feature_idx] = mult * blob.data[example_i, feature_idx]
        elif len(blob.data.shape) == 4:
            spatial_max_idx = blob.data[example_i, feature_idx].argmax()
            row, col = np.unravel_index(spatial_max_idx, blob.data.shape[2:])
            blob.diff[example_i, feature_idx, row, col] = mult * blob.data[example_i, feature_idx, row, col]
        else:
            raise Exception('source/target blobs should be shaped as ' \
                            'if from a conv/fc layer')

    def _weight_child_neurons(self, path, bottom_id, num_children):
        '''
        Weight bottom blob neurons with respect to top blob activations.

        Returns a 1d numpy array with one entry for each feature in the bottom blob.
        Higher values indicate the top blob is more strongly connected to that feature.
        '''
        self._backprop_path(path)

        layers = self.config['layers']
        top_id = self._node_to_layer_id(path[-1])
        top_layer_name = layers[top_id]['layer_name']
        bottom_layer_name = layers[bottom_id]['layer_name']
        bottom_blob_name = layers[bottom_id]['blob_name']
        self.net.backward(start=top_layer_name, end=bottom_layer_name)
        bottom_blob = self.net.blobs[bottom_blob_name]

        # compute weights between neurons (backward is needed to do deconvolution on the conv layers)
        edge_weights = bottom_blob.data[0] * bottom_blob.diff[0]
        if len(edge_weights.shape) == 3:
            edge_weights = edge_weights.mean(axis=(1, 2))
        assert len(edge_weights.shape) == 1
        return abs(edge_weights)

    # Image manipulation

    def _set_image(self, img_fname):
        # remove alpha channel if present
        img = io.imread(img_fname)[:, :, :3]
        img = img_as_ubyte(trans.resize(img, [227, 227]))
        img = self._unshowable(img)
        self._replicate_first_image(img)

    def _unshowable(self, img):
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] - self.mean
        img = img.transpose([2, 0, 1])
        return img

    def _replicate_first_image(self, img=None):
        img_blob = self.net.blobs[self.config['image_blob_name']]
        if img is not None:
            img_blob.data[0] = img
        for i in range(1, img_blob.data.shape[0]):
            img_blob.data[i] = img_blob.data[0]

    def _showable(self, img, rescale=False):
        # NOTE: assumes images in the net are BGR
        img = img.transpose([1, 2, 0])
        img = (img + self.mean)[:, :, ::-1]
        if rescale and (img.min() < 0 or 255 < img.max()):
            img = rescale_intensity(img)
        img = img.clip(0, 255).astype(np.uint8)
        return img

    # caffe stuff

    @property
    def net(self):
        if not hasattr(self, '_net'):
            self._net = self._load_net(self.net_param)
        return self._net

    def _load_param(self, with_data=False):
        if with_data:
            spec_fname = self.config.spec_wdata
        else:
            spec_fname = self.config.spec_nodata
        net_param = cpb.NetParameter()
        with open(spec_fname, 'r') as f:
            text_format.Merge(f.read(), net_param)
        return net_param

    def _load_net(self, net_param):
        '''
        Takes a network spec file and returns a NamedTemporaryFile which
        contains the modified spec with ReLUs appropriate for visualization.
        '''
        net_param = _convert_relus(net_param, relu_type=self.config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        if self.gpu_id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)


########################
# These build the cached feature grids.
# First an lmdb of gradient visualizations is computed (build_max_act_db)
# then they're combined into 'canonical images', i.e., grids of image patchs
# and the corresponding ZF style vis patches.
# TODO: refactor... lots of this is redundant with VisTree

class FeatBuilder(object):

    def __init__(self, config, gpu_id):
        self.config = config
        self.gpu_id = gpu_id

    @staticmethod
    def _get_key(blob_name, feature_idx):
        return '{}_{}'.format(blob_name, feature_idx)

    def _load_param(self, with_data=False):
        if with_data:
            spec_fname = self.config.spec_wdata
        else:
            spec_fname = self.config.spec_nodata
        net_param = cpb.NetParameter()
        with open(spec_fname, 'r') as f:
            text_format.Merge(f.read(), net_param)
        return net_param

    def _load_net(self, net_param):
        '''
        Takes a network spec file and returns a NamedTemporaryFile which
        contains the modified spec with ReLUs appropriate for visualization.
        '''
        net_param = _convert_relus(net_param, relu_type=self.config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        if self.gpu_id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)

    def num_features(self, blob_name):
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        blob = net.blobs[blob_name]
        return blob.data.shape[1]

    def build_max_act_db(self, blob_name, k=5):
        # don't use self.net, which a deploy net (data comes from python)

        def _get_blob_layer(net_param, blob_name):
            '''
            Return the name of the last layer to output a blob.
            '''
            layer_name = None
            for layer in net_param.layer:
                if blob_name in layer.top:
                    layer_name = layer.name
            if layer_name == None:
                raise Exception('could not find a layer that outputs ' \
                                'blob {}'.format(blob_name))
            return layer_name

        def _len_lmdb(fname):
            logger.info('computing lmdb size...')
            db = lmdb.open(fname)
            n_examples = sum(1 for _ in db.begin().cursor())
            del db
            logger.info('found lmdb size: {} examples'.format(n_examples))
            return n_examples

        def _showable(img, rescale=False):
            # NOTE: assumes images in the net are BGR
            img = img.transpose([1, 2, 0])
            img = (img + mean)[:, :, ::-1]
            img = img.clip(0, 255).astype(np.uint8)
            if rescale:
                img = rescale_intensity(img)
            return img

        def _reconstruct_backward(net, net_param, blob_name, blob_idxs, act_vals=None, act_mult=1):
            blob = net.blobs[blob_name]
            blob.diff[:] = 0
            for i, blob_idx in enumerate(blob_idxs):
                if act_vals == None:
                    blob.diff[blob_idx] = act_mult * blob.data[blob_idx]
                else:
                    blob.diff[blob_idx] = act_mult * act_vals[i]
            layer_name = _get_blob_layer(net_param, blob_name)
            net.backward(start=layer_name, end=img_layer_name)

        def _to_bbox(img, blob_idx):
            '''
            Take an image which is mostly 0s and return the smallest
            bounding box which contains all non-0 entries.

            img     array of size (c, h, w)
            blob_idx    tuple with (num, channels, height, width) or (num, channels)
            '''
            # (num, channel, height, width)
            if len(blob_idx) == 4:
                row, col = blob_idx[-2:]
                m = abs(img).max(axis=0)
                linear_idx_map = np.arange(np.prod(m.shape)).reshape(m.shape)
                linear_idxs = linear_idx_map[m > 0]
                rows = (linear_idxs // m.shape[0])
                cols = (linear_idxs % m.shape[0])
                if np.prod(rows.shape) == 0 or np.prod(cols.shape) == 0:
                    top_left = row, col
                    bottom_right = row, col
                else:
                    top_left = (rows.min(), cols.min())
                    bottom_right = (rows.max(), cols.max())
                return (top_left, bottom_right)
            # (num, channel)
            elif len(blob_idx) == 2:
                return ((0, 0), (img.shape[1]-1, img.shape[2]-1))
            else:
                raise Exception('do not know how to create a bounding box from ' \
                                'blob_idx {}'.format(blob_idx))

        mean = load_mean_image(self.config.mean_fname)
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        img_layer_name = _get_blob_layer(data_net_param, self.config.image_blob_name)
        batch_size = self.config.batch_size
        n_batches = int(math.ceil(self.config.num_examples / float(batch_size)))
        layers = {l.name: l for l in data_net_param.layer}
        dbname = layers[self.config.data_layer_name].data_param.source
        n_db_examples = _len_lmdb(dbname)
        assert n_db_examples == self.config.num_examples
        assert n_db_examples % batch_size == 0
        example_offset = 0
        # TODO: load the top_k lists from the db and append to them if they already exist
        maxes = defaultdict(lambda: [{'activation': -np.inf} for _ in range(k)])
        logger.info('blob {}'.format(blob_name))
        img_blob = net.blobs[self.config.image_blob_name]
        blob = net.blobs[blob_name]
        assert blob.data.shape[0] == batch_size
        n_features = blob.data.shape[1]

        for batch_i in xrange(n_batches):
            net.forward()
            logger.info('batch {}'.format(batch_i))

            # for each example
            for num in xrange(batch_size): # examples
                logger.info('example {}'.format(example_offset + num))
                for chan in xrange(n_features): # channel
                    key = self._get_key(blob_name, chan)
                    top_k = maxes[key]
                    fmap_idx = blob.data[num, chan].argmax()
                    if len(blob.data.shape) > 2:
                        fmap_idx = np.unravel_index(fmap_idx, blob.data.shape[2:])
                    else:
                        fmap_idx = tuple()
                    blob_idx = (num, chan) + fmap_idx
                    act = blob.data[blob_idx]

                    # is this example's best patch in the topk?
                    if act > top_k[0]['activation']:
                        img = img_blob.data[num, :, :, :]
                        entry = {
                            'example_idx': example_offset + num,
                            'feature_map_loc': blob_idx[2:] if len(blob_idx) > 2 else tuple(),
                            'img': _showable(img),
                            #'reconstruction': self._showable(reconstruction),
                            'reconstruct_idx': blob_idx,
                            'batch_idx': batch_i,
                            'num': num,
                            'activation': act,
                            #'patch_bbox': bbox,
                        }
                        top_k_idx = bisect([v['activation'] for v in top_k], act)
                        top_k.insert(top_k_idx, entry)
                        del top_k[0]

            example_offset += batch_size

        entries_per_example = [[] for _ in range(n_db_examples)]
        for chan in range(n_features):
            key = self._get_key(blob_name, chan)
            top_k = maxes[key]
            for entry in top_k:
                max_act = top_k[-1]['activation']
                ex_idx = entry['example_idx']
                entries_per_example[ex_idx].append((entry['reconstruct_idx'], max_act, entry))

        # for each example, list the reconstructions which must be computed

        example_offset = 0
        # compute those reconstructions
        for batch_i in xrange(n_batches):
            net.forward()
            logger.info('rec batch {}'.format(batch_i))

            entries_in_batch  = entries_per_example[example_offset:example_offset+batch_size]
            def total_entries():
                total = 0
                return sum(len(e) for e in entries_in_batch)

            while total_entries():
                entries_to_process = [ent.pop() for ent in entries_in_batch if ent]
                blob_idxs, act_vals, blob_entries = zip(*entries_to_process)
                # one idx per example
                assert len(set(zip(*blob_idxs)[0])) == len(entries_to_process)
                _reconstruct_backward(net,
                                      data_net_param,
                                      blob_name,
                                      blob_idxs,
                                      act_vals,
                                      act_mult=self.config.blob_multipliers[blob_name])
                for blob_idx, entry in zip(blob_idxs, blob_entries):
                    num = entry['num']
                    reconstruction = img_blob.diff[num, :, :, :]
                    bbox = _to_bbox(reconstruction, blob_idx)
                    entry['reconstruction'] = _showable(reconstruction, rescale=True)
                    entry['patch_bbox'] = bbox

            example_offset += batch_size


        logger.info('finished computing maximum activations... writing to db')
        act_env = lmdb.open(self.config.max_activation_dbname, map_size=recon.config.config.lmdb_map_size)
        with act_env.begin(write=True) as txn:
            for key, top_k in maxes.iteritems():
                s = pkl.dumps(top_k)
                txn.put(key, s)


    def canonical_image(self, blob_name, feature_idx, k, tmp_fname):
        # load the image and reconstruction
        act_env = lmdb.open(self.config.max_activation_dbname, map_size=recon.config.config.lmdb_map_size)
        act_key = self._get_key(blob_name, feature_idx)
        with act_env.begin() as txn:
            val = txn.get(act_key)
            if val == None:
                raise Exception('activation for key {} not yet stored'.format(act_key))
            activations = pkl.loads(val)[-k:]
        img_patches = []
        rec_patches = []

        # crop patches from the image and reconstruction
        for i, act in enumerate(activations):
            rec = act['reconstruction']
            img = act['img']
            top_left, bottom_right = act['patch_bbox']
            top, left, bottom, right = top_left + bottom_right
            img = img[top:bottom+1, left:right+1]
            rec = rec[top:bottom+1, left:right+1]
            img_patches.append(img)
            rec_patches.append(rec)

        # display the patches in a grid, then save to a file
        patch_size = [0, 0]
        for img, rec in zip(img_patches, rec_patches):
            patch_size[0] = max(img.shape[0], patch_size[0])
            patch_size[0] = max(rec.shape[0], patch_size[0])
            patch_size[1] = max(img.shape[1], patch_size[1])
            patch_size[1] = max(rec.shape[1], patch_size[1])
        def scale(patch):
            assert len(patch.shape) == 3
            if patch.shape[:2] == patch_size:
                return patch
            new_patch = np.zeros(tuple(patch_size) + patch.shape[2:3], dtype=patch.dtype)
            new_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch[:, :, :]
            return new_patch
        img_patches = (scale(img) for img in img_patches)
        rec_patches = (scale(rec) for rec in rec_patches)
        pad_img_patches = [np.pad(img, ((0, 2), (0, 2), (0, 0)), 'constant')
                                                for img in img_patches]
        pad_rec_patches = [np.pad(rec, ((0, 2), (0, 2), (0, 0)), 'constant')
                                                for rec in rec_patches]
        img_patches = np.vstack([img[np.newaxis] for img in pad_img_patches])
        rec_patches = np.vstack([rec[np.newaxis] for rec in pad_rec_patches])
        img_mon_channels = []
        rec_mon_channels = []
        for channel in range(img_patches.shape[-1]):
            imgs = img_patches[:, :, :, channel]
            mon = montage.montage2d(imgs, fill=0, rescale_intensity=False)
            img_mon_channels.append(mon)
            recs = rec_patches[:, :, :, channel]
            mon = montage.montage2d(recs, fill=0, rescale_intensity=False)
            rec_mon_channels.append(mon)
        img = np.dstack(img_mon_channels)
        rec = np.dstack(rec_mon_channels)
        combined = np.hstack([img, np.zeros([img.shape[0], 5, img.shape[2]], dtype=img.dtype), rec])
        io.imsave(tmp_fname, combined)

