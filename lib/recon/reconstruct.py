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

from recon.config import config, relu_backward_types
from recon.util import load_mean_image

import logging
logger = logging.getLogger(config.logger.name)

# main api calls

def canonical_image(net_id, blob_name, feature_idx, k):
    net = nets[net_id]
    net.canonical_image(blob_name, feature_idx, k)



class VisTree(object):

    def __init__(self, net_id, img_fname):
        self.net_id = net_id
        self.config = config.nets[self.net_id]
        self.net_param = self._load_param(with_data=False)
        self.image_blob = 'data'
        self.mean = load_mean_image(self.config.mean_fname)
        self._set_image(img_fname)
        self.net.forward()
        self.dag = nx.DiGraph()
        # TODO: remove these
        self.prev_layer_map = {
            # TODO: make these work
            'fc8': 'fc7',
            'fc7': 'fc6',
            'fc6': 'conv5',
            'conv5': 'conv4',
            'conv4': 'conv3',
            'conv3': 'conv2',
            'conv2': 'conv1',
            'conv1': 'data',
        }
        self._reconstructions = {}

    def image(self):
        img_blob = self.net.blobs[self.image_blob]
        return self._showable(img_blob.data[0])

    def _set_image(self, img_fname):
        img = io.imread(img_fname)
        img = img_as_ubyte(trans.resize(img, [227, 227]))
        img = self._unshowable(img)
        self._replicate_first_image(img)

    def _unshowable(self, img):
        img = img.astype(np.uint8)
        img = img[:, :, ::-1] - self.mean
        img = img.transpose([2, 0, 1])
        return img

    def _replicate_first_image(self, img=None):
        img_blob = self.net.blobs[self.image_blob]
        if img is not None:
            img_blob.data[0] = img
        for i in range(1, img_blob.data.shape[0]):
            img_blob.data[i] = img_blob.data[0]


    def _node_name(self, blob_path, act_ids):
        return '-'.join([blob + '_' + str(act_id) for blob, act_id in zip(blob_path, act_ids)])


    def reconstruction(self, layer_path, feature_paths):
        '''
        Visualize the given blob/feature pairs in the deconv fashion.
        All backprop happens from layer_name to the input. Starting backprop from multiple
        is not yet supported.

        If `paths` is specified then, for each item, it should specify a list of blob names
        from root to this node. Only gradient information from those blobs will
        be backpropped. TODO: better explain

        ZF vis is a special case where layer_path = [top_layer]

        Return a list of dicts, each with a 'reconstruction' and 'bbox' key.
        '''
        blob_path = [self.config['layer_to_blob'][layer] for layer in layer_path]

        # zero out everything but the max pixel
        nodes = []
        nodes_need_backward = []
        filtered_feature_paths = []
        for example_i, feature_path in enumerate(feature_paths):
            node_name = self._node_name(blob_path, feature_path)
            nodes.append(node_name)
            blob = self.net.blobs[blob_path[0]]
            if node_name in self._reconstructions:
                continue
            if len(feature_path) > blob.data.shape[0]:
                raise Exception('Currently the number of visualized examples must be at most' \
                                'the batch size of the network.')
            nodes_need_backward.append(node_name)
            filtered_feature_paths.append(feature_path)

        def filter_feature(example_i, feature_idx, blob_name):
            blob = self.net.blobs[blob_name]
            img = blob.diff[example_i]
            total = abs(img).sum()
            total_feature = abs(img[feature_idx]).sum()
            mult = total / total_feature
            assert mult >= 1.0
            blob.diff[example_i, :feature_idx] = 0
            blob.diff[example_i, feature_idx+1:] = 0
            blob.diff[example_i] *= mult

        def set_max_pixel(example_i, feature_idx, blob_name):
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

        # backprop, re-focusing on particular features at each step
        assert layer_path[-1] != self.image_blob
        layer_path += [None]
        inv_feature_paths = zip(*filtered_feature_paths)
        for layer_i, tup in enumerate(zip(layer_path[:-1], layer_path[1:], inv_feature_paths, blob_path)):
            top_layer, bottom_layer, feature_idxs, top_blob_name = tup
            # filter activations along feature paths
            for example_i, feature_idx in enumerate(feature_idxs):
                if layer_i == 0:
                    set_max_pixel(example_i, feature_idx, top_blob_name)
                else:
                    filter_feature(example_i, feature_idx, top_blob_name)
            if bottom_layer is None:
                self.net.backward(start=top_layer)
            else:
                self.net.backward(start=top_layer, end=bottom_layer)

        # cache the visualization
        for example_i, node_name in enumerate(nodes_need_backward):
            img_blob = self.net.blobs[self.image_blob]
            reconstruction = np.copy(img_blob.diff[example_i, :, :, :])
            # TODO: add bounding box back in
            #bbox = self._to_bbox(reconstruction)
            reconstruction = self._showable(reconstruction)
            # TODO: it might be a good idea to put this in the graph, but for now not, because it's not JSON serializable
            self._reconstructions[node_name] = {
                'reconstruction': reconstruction,
                #'bbox': bbox,
            }

        return [self._reconstructions[node] for node in nodes]


    def max_idxs(self, layer_id):
        '''
        Return a list of feature indices which maximally activate the layer.
        '''
        blob_name = self.config['layers'][layer_id]['blob_name']
        blob = self.net.blobs[blob_name]
        if len(blob.data.shape) == 2:
            features = blob.data
        elif len(blob.data.shape) == 4:
            features = blob.data.max(axis=(2, 3))
        return list(features[0].argsort()[::-1])


    def tree(self, top_layer_id, act_id):
        '''
        Return a JSON serializable dictionary representing a hierarchy of
        features with root given by the top layer and activation id.
        '''
        top_layer_name = self.config['layers'][top_layer_id]['layer_name']
        top_blob_name = self.config['layers'][top_layer_id]['blob_name']
        bottom_layer_id = self.prev_layer_map[top_layer_id]
        bottom_layer_name = self.config['layers'][bottom_layer_id]['layer_name']
        bottom_blob_name = self.config['layers'][bottom_layer_id]['blob_name']

        root = self._expand(top_layer_name, top_blob_name,
                           bottom_layer_name, bottom_blob_name,
                           act_id)
        # TODO: don't allow arbitrary depth tree to be returned
        successor_tree = nx.ego_graph(self.dag, root, radius=len(self.dag))
        tree_dict = json_graph.tree_data(successor_tree, root, attrs={'children': 'children', 'id': 'name'})
        return tree_dict


    def expand(self, top_layer_id, act_id, num_children=5):
        '''
        Grow the tree by exanding the top num_children nodes from the given activation.
        '''
        top_layer_name = self.config['layers'][top_layer_id]['layer_name']
        top_blob_name = self.config['layers'][top_layer_id]['blob_name']
        bottom_layer_id = self.prev_layer_map[top_layer_id]
        bottom_layer_name = self.config['layers'][bottom_layer_id]['layer_name']
        bottom_blob_name = self.config['layers'][bottom_layer_id]['blob_name']

        root, expanded_nodes = self._expand(top_layer_name, top_blob_name,
                     bottom_layer_name, bottom_blob_name, act_id, num_children, return_expanded=True)
        expanded_data = [self.dag.node[node] for node in expanded_nodes]
        return expanded_data


    def _compute_weights(self, top_layer_name, top_blob_name,
                      bottom_layer_name, bottom_blob_name, act_id,
                      num_children):
        '''
        Compute weights between the given top and bottom layers.

        Returns a 1d numpy array with one entry for each feature in the bottom blob.
        Higher values indicate the top blob is more strongly connected to that feature.
        '''
        bottom_blob = self.net.blobs[bottom_blob_name]
        top_blob = self.net.blobs[top_blob_name]
        img_blob = self.net.blobs[self.image_blob]

        # set all but 1 pixel of the top diff to 0
        top_blob.diff[0] = 0
        if len(top_blob.data.shape) == 2:
            top_blob.diff[0, act_id] = 1.0
        elif len(top_blob.data.shape) == 4:
            spatial_max_idx = top_blob.data[0, act_id].argmax()
            row, col = np.unravel_index(spatial_max_idx, top_blob.data.shape[2:])
            top_blob.diff[0, act_id, row, col] = 1.0
        else:
            raise Exception('source/target blobs should be shaped as ' \
                            'if from a conv/fc layer')
        self.net.backward(start=top_layer_name, end=bottom_layer_name)

        # compute weights between neurons (backward is needed to do deconvolution on the conv layers)
        edge_weights = bottom_blob.data[0] * bottom_blob.diff[0]
        if len(edge_weights.shape) == 3:
            edge_weights = edge_weights.mean(axis=(1, 2))
        assert len(edge_weights.shape) == 1
        return abs(edge_weights)


    def _expand(self, top_layer_name, top_blob_name,
                bottom_layer_name, bottom_blob_name, act_id,
                num_children=5, return_expanded=False):
        '''
        Compute weights between layers w.r.t. a particular activation.

        Given a pair of layers and an activation to focus on in the
        top layer, compute the edge weights between the activation
        and all activations of the previous layer.
        '''
        # compute weights
        edge_weights = self._compute_weights(top_layer_name, top_blob_name,
                bottom_layer_name, bottom_blob_name, act_id, num_children)

        # reconstruct strong connections
        important_bottom_idxs = edge_weights.argsort()[::-1][:num_children]
        recons = self.reconstruction([bottom_layer_name], [[i] for i in important_bottom_idxs])

        # fill in the dag with edges from top to bottom and meta data
        dag = self.dag
        img_blob = self.net.blobs[self.image_blob]
        top_node = self._node_name([top_blob_name], [act_id])
        expanded_nodes = []

        dag.add_node(top_node)
        dag.node[top_node]['blob_name'] = top_blob_name
        dag.node[top_node]['act_id'] = act_id
        for k in range(num_children):
            bottom_idx = important_bottom_idxs[k]
            bottom_node = self._node_name([bottom_blob_name], [bottom_idx])

            dag.add_edge(top_node, bottom_node, attr_dict={
                'weight': edge_weights[bottom_idx],
            })
            dag.node[bottom_node]['blob_name'] = bottom_blob_name
            dag.node[bottom_node]['act_id'] = bottom_idx

            expanded_nodes.append(bottom_node)

        # return the node which was expanded
        if return_expanded:
            return top_node, expanded_nodes
        else:
            return top_node


    def _showable(self, img, rescale=False):
        # TODO: don't always assume images in the net are BGR
        img = img.transpose([1, 2, 0])
        img = (img + self.mean)[:, :, ::-1]
        img = img.clip(0, 255).astype(np.uint8)
        if rescale:
            img = rescale_intensity(img)
        return img

    def _to_bbox(self, img):
        '''
        Take an image which is mostly 0s and return the smallest
        bounding box which contains all non-0 entries.

        img     array of size (c, h, w)
        '''
        # (num, channel, height, width)
        # TODO: don't always assume the reconstruction comes from a (4-tensor) conv layer
        if True: #len(blob_idx) == 4:
            #row, col = blob_idx[-2:]
            m = abs(img).max(axis=0)
            linear_idx_map = np.arange(np.prod(m.shape)).reshape(m.shape)
            linear_idxs = linear_idx_map[m > 0]
            rows = (linear_idxs // m.shape[0])
            cols = (linear_idxs % m.shape[0])
            if np.prod(rows.shape) == 0 or np.prod(cols.shape) == 0:
                raise Exception('TODO: not supported for now')
                #top_left = row, col
                #bottom_right = row, col
            else:
                top_left = (rows.min(), cols.min())
                bottom_right = (rows.max(), cols.max())
            return (top_left, bottom_right)
        # (num, channel)
        #elif len(blob_idx) == 2:
        #    return ((0, 0), (img.shape[1]-1, img.shape[2]-1))
        else:
            raise Exception('do not know how to create a bounding box from ' \
                            'blob_idx {}'.format(blob_idx))


    @staticmethod
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
        # TODO: also accept file objects instead of just names?
        net_param = Reconstructor._convert_relus(net_param, relu_type=self.config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)



class Reconstructor(object):

    @staticmethod
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

    def __init__(self, net_id):
        self.net_id = net_id
        self.config = config.nets[self.net_id]
        self.act_env = lmdb.open(self.config.max_activation_dbname, 
                                 map_size=config.lmdb_map_size)
        self.mean = load_mean_image(self.config.mean_fname)
        self.net_param = self._load_param(with_data=True)

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
        # TODO: also accept file objects instead of just names?
        net_param = Reconstructor._convert_relus(net_param, relu_type=self.config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)

    def num_features(self, blob_name):
        return self.net.blobs[blob_name].data.shape[1]

    def _get_blob_layer(self, net_param, blob_name):
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

    def _get_key(self, blob_name, feature_idx):
        return '{}_{}'.format(blob_name, feature_idx)

    def _showable(self, img, rescale=False):
        # TODO: don't always assume images in the net are BGR
        img = img.transpose([1, 2, 0])
        img = (img + self.mean)[:, :, ::-1]
        img = img.clip(0, 255).astype(np.uint8)
        if rescale:
            img = rescale_intensity(img)
        return img

    def _to_bbox(self, img, blob_idx):
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

    def _reconstruct_backward(self, net, net_param, blob_name, blob_idxs, act_vals=None, act_mult=1):
        blob = net.blobs[blob_name]
        blob.diff[:] = 0
        for i, blob_idx in enumerate(blob_idxs):
            if act_vals == None:
                blob.diff[blob_idx] = act_mult * blob.data[blob_idx]
            else:
                blob.diff[blob_idx] = act_mult * act_vals[i]
        layer_name = self._get_blob_layer(net_param, blob_name)
        net.backward(start=layer_name, end=self.img_layer_name)

    def _len_lmdb(self, fname):
        logger.info('computing lmdb size...')
        db = lmdb.open(fname)
        n_examples = sum(1 for _ in db.begin().cursor())
        del db
        logger.info('found lmdb size: {} examples'.format(n_examples))
        return n_examples

    def max_idxs(self, blob_name, k=5):
        net = self.net
        net.forward()
        spatial_max = net.blobs[blob_name].data.max(axis=(2, 3))
        return spatial_max[0].argsort()[::-1][:k]

    def build_max_act_db(self, blob_name, k=5):
        # don't use self.net, which a deploy net (data comes from python)
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        self.img_layer_name = self._get_blob_layer(data_net_param, 
                                                   self.config.img_blob_name)
        batch_size = self.config.batch_size
        n_batches = int(math.ceil(self.config.num_examples / float(batch_size)))
        layers = {l.name: l for l in data_net_param.layer}
        dbname = layers[self.config.data_layer_name].data_param.source
        n_db_examples = self._len_lmdb(dbname)
        assert n_db_examples == self.config.num_examples
        assert n_db_examples % batch_size == 0
        example_offset = 0
        # TODO: load the top_k lists from the db and append to them if they already exist
        maxes = defaultdict(lambda: [{'activation': -np.inf} for _ in range(k)])
        logger.info('blob {}'.format(blob_name))
        img_blob = net.blobs[self.config.img_blob_name]
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
                            'img': self._showable(img),
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
                self._reconstruct_backward(net,
                                           data_net_param,
                                           blob_name,
                                           blob_idxs,
                                           act_vals,
                                           act_mult=self.config.blob_multipliers[blob_name])
                for blob_idx, entry in zip(blob_idxs, blob_entries):
                    num = entry['num']
                    reconstruction = img_blob.diff[num, :, :, :]
                    bbox = self._to_bbox(reconstruction, blob_idx)
                    entry['reconstruction'] = self._showable(reconstruction, rescale=True)
                    entry['patch_bbox'] = bbox

            example_offset += batch_size


        logger.info('finished computing maximum activations... writing to db')
        with self.act_env.begin(write=True) as txn:
            for key, top_k in maxes.iteritems():
                s = pkl.dumps(top_k)
                txn.put(key, s)


    def canonical_image(self, blob_name, feature_idx, k, tmp_fname):
        act_key = self._get_key(blob_name, feature_idx)
        with self.act_env.begin() as txn:
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

        # display the patches in a grid
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

    def reconstruct(self, blob_names):
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        self.img_layer_name = self._get_blob_layer(data_net_param, 
                                                   self.config.img_blob_name)
        for _ in range(8):
            net.forward()
        img_id = 8

        img_blob = net.blobs[self.config.img_blob_name]
        io.imsave('/tmp/img.jpg', self._showable(img_blob.data[img_id]))

        for blob_name in blob_names:
            blob = net.blobs[blob_name]
            logger.info('single image blob {}'.format(blob_name))

            # fc
            if len(blob.data.shape) == 2:
                top_features = (-blob.data[img_id]).argsort()
                blob_idxs = [(img_id, idx) for idx in top_features]
            # conv
            elif len(blob.data.shape) == 4:
                patch_idxs = []
                patch_maxes = []
                for feat_i in range(blob.data.shape[1]):
                    patch_idx = blob.data[img_id, feat_i].argmax()
                    patch_idx = np.unravel_index(patch_idx, blob.data.shape[2:])
                    patch_idxs.append((img_id, feat_i) + patch_idx)
                    patch_maxes.append(blob.data[img_id, feat_i].max())
                top_features = np.array(patch_maxes).argsort()[::-1]
                blob_idxs = [patch_idxs[i] for i in top_features]
            else:
                raise Exception('only fc or conv blobs supported, not {}'.format(blob_name))

            for i, blob_idx in enumerate(blob_idxs[:40]):
                self._reconstruct_backward(net,
                                           data_net_param,
                                           blob_name,
                                           [blob_idx],
                                           act_mult=self.config.blob_multipliers[blob_name])
                io.imsave('/tmp/recon_{}_ord{}_feat{}.jpg'.format(blob_name, i, top_features[i]),
                          self._showable(img_blob.diff[img_id]))

    def graph(self):
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        layer_names = list(net._layer_names)
        logger.info('forwarding')
        for _ in range(8):
            net.forward()
        img_id = 8

        # self.layer_edges is a list of (layer_source, layer_target) pairs
        # (layer names) between which edge values should be computed.

        nodes = []
        links = []
        adj_mats = []

        # edges are in the direction of forward prop
        for edge in self.config.edges:
            # top
            target_blob = net.blobs[edge.target_blob]
            source_blob = net.blobs[edge.source_blob]
            adj_mat = np.zeros([target_blob.data.shape[1], source_blob.data.shape[1]])
            adj_mats.append(adj_mat)
            for feat_i in range(source_blob.data.shape[1]):
                source_name = edge.source_blob + '_{}'.format(feat_i)
                nodes.append(source_name)
            for target_feat_i in range(target_blob.data.shape[1]):
                target_name = edge.target_blob + '_{}'.format(target_feat_i)
                logger.info('target: {}'.format(target_name))
                nodes.append(target_name)
                target_blob.diff[:] = 0
                if len(target_blob.data.shape) == 2:
                    target_blob.diff[img_id, target_feat_i] = 1.0
                elif len(target_blob.data.shape) == 4:
                    act = target_blob.data[img_id, target_feat_i]
                    max_act = act.max()
                    max_idx = act.argmax()
                    max_idx_2, max_idx_3 = np.unravel_index(max_idx, act.shape)
                    target_blob.diff[img_id, target_feat_i, max_idx_2, max_idx_3] = 1.0
                else:
                    raise Exception('source/target blobs should be shaped as ' \
                                    'if from a conv/fc layer')
                # At least for ReLUs, backward is the same as a linear backward pass
                # of the hidden activations.
                net.backward(start=edge.target_layer, end=edge.source_layer)
                edge_weights = source_blob.data * source_blob.diff
                if len(edge_weights.shape) == 4:
                    edge_weights = edge_weights.sum(axis=(2, 3))
                for source_feat_i in range(source_blob.data.shape[1]):
                    source_name = edge.source_blob + '_{}'.format(source_feat_i)
                    weight = edge_weights[img_id, source_feat_i]
                    links.append((source_name, target_name, weight))
                    adj_mat[target_feat_i, source_feat_i] = weight

        return {
            'nodes': nodes,
            'links': links,
            'adj_mats': adj_mats,
        }

