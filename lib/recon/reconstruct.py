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

# main api calls


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
    Keep track of explored reconstructions and allow efficient exploration of the space.
    Try to answer the question "How does this network interpret this image?"

    This actually can keep track of a forest of visualizations. Each tree is rooted at
    a particular ZF style gradient vis. New nodes are created by expanding an existing
    node. This creates some children which visualize gradients backpropped from the root,
    but with the feature map blobs of the children further masked similar to how the
    root feature map is masked. Keeping track of the nodes in this tree allows visualizations
    to be computed efficiently in batches instead of one at a time.

    NOTE: ZF means Zeiler/Fergus in reference to "Visualizing and Understanding Convolutional Networks"
    '''

    def __init__(self, net_id, img_fname):
        '''
        # Args
            net_id: Network to inspect (lib/recon/config.py)
            img_fname: Image to inspect (data/gallery/)
        '''
        self.net_id = net_id
        self.config = config.nets[self.net_id]
        self.net_param = self._load_param(with_data=False)
        self.image_blob = 'data'
        self.prob_blob = 'prob'
        self.mean = load_mean_image(self.config.mean_fname)
        self._labels = load_ilsvrc12_labels(self.config.labels_fname)
        self._set_image(img_fname)
        self.net.forward()
        self.dag = nx.DiGraph()
        self._reconstructions = {}

    # Exposed api

    def labels(self, top_k=5):
        '''
        Return a list of the top k labels assigned to the image

        # Args
            top_k: Number of labels to return, with most likely first
        '''
        prob = self.net.blobs[self.prob_blob].data[0].flatten()
        top_idxs = prob.argsort()[::-1][:top_k]
        template = '{} ({:.2f}, {})'
        return [template.format(self._labels[i], prob[i], i) for i in top_idxs]

    def image(self):
        '''
        Return the image visualized by this net
        '''
        img_blob = self.net.blobs[self.image_blob]
        return self._showable(img_blob.data[0])

    def reconstruction(self, path):
        '''
        Visualize the given blob/feature pairs in the deconv fashion with layers
        specified in `path` masked to specific features.

        

        ZF vis is a special case where layer_path = [top_layer]

        Return a list of dicts, each with a 'reconstruction' and 'bbox' key.
        '''
        # TODO: temporary to test new api
        return self.image()
        blob_path = [self.layer_to_blob[layer] for layer in layer_path]
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

        # backprop, re-focusing on particular features at each step
        assert layer_path[-1] != self.image_blob
        layer_path += [None]
        inv_feature_paths = zip(*filtered_feature_paths)
        for layer_i, tup in enumerate(zip(layer_path[:-1], layer_path[1:], inv_feature_paths, blob_path)):
            top_layer, bottom_layer, feature_idxs, top_blob_name = tup
            # filter activations along feature paths
            for example_i, feature_idx in enumerate(feature_idxs):
                if layer_i == 0:
                    self.set_max_pixel(example_i, feature_idx, top_blob_name)
                else:
                    self.filter_feature(example_i, feature_idx, top_blob_name)
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

    def max_blob_idxs(self, blob_name):
        '''
        Return a list of feature indices which maximally activate the blob.
        '''
        blob = self.net.blobs[blob_name]
        if len(blob.data.shape) == 2:
            features = blob.data
        elif len(blob.data.shape) == 4:
            # TODO: might want to do this in different ways
            features = blob.data.max(axis=(2, 3))
        print features.shape
        return list(features[0].argsort()[::-1])

    def children_from_path(self, path, num_children=5):
        #max_nodes = self._max_children(path)
        # TODO
        return []


    # Helpers

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
        img_blob = self.net.blobs[self.image_blob]
        if img is not None:
            img_blob.data[0] = img
        for i in range(1, img_blob.data.shape[0]):
            img_blob.data[i] = img_blob.data[0]

    def _showable(self, img, rescale=False):
        # TODO: don't always assume images in the net are BGR
        img = img.transpose([1, 2, 0])
        img = (img + self.mean)[:, :, ::-1]
        if rescale and (img.min() < 0 or 255 < img.max()):
            img = rescale_intensity(img)
        img = img.clip(0, 255).astype(np.uint8)
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

    # TODO
    def _max_children(self, path):
        pass

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
        net_param = _convert_relus(net_param, relu_type=self.config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)

    @property
    def layer_to_blob(self):
        if not hasattr(self, '_layer_to_blob'):
            self._layer_to_blob = { l.layer_name: l.blob_name for l in self.config['layers'] }
        return self._layer_to_blob




    
    # Extras (TODO)





    def _node_name(self, blob_path, act_ids):
        return '-'.join([blob + '_' + str(act_id) for blob, act_id in zip(blob_path, act_ids)])

    def filter_feature(self, example_i, feature_idx, blob_name):
        blob = self.net.blobs[blob_name]
        img = blob.diff[example_i]
        total = abs(img).sum()
        total_feature = abs(img[feature_idx]).sum()
        mult = total / total_feature
        #mult = self.config.blob_multipliers[blob_name]
        assert mult >= 1.0
        blob.diff[example_i, :feature_idx] = 0
        blob.diff[example_i, feature_idx+1:] = 0
        #blob.diff[example_i] *= 20 * mult / total_feature
        blob.diff[example_i] *= mult


    def set_max_pixel(self, example_i, feature_idx, blob_name):
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

        dag = self.dag
        top_node = self._node_name([top_blob_name], [act_id])
        layer_path = dag.node[top_node]['path'] + [top_layer_name, bottom_layer_name]
        path_ids = dag.node[top_node]['path_ids'] + [act_id]
        blob_path = [self.layer_to_blob[layer] for layer in layer_path]

        ## set all but 1 pixel of the top diff to 0
        #top_blob.diff[0] = 0
        #if len(top_blob.data.shape) == 2:
        #    top_blob.diff[0, act_id] = 1.0
        #elif len(top_blob.data.shape) == 4:
        #    spatial_max_idx = top_blob.data[0, act_id].argmax()
        #    row, col = np.unravel_index(spatial_max_idx, top_blob.data.shape[2:])
        #    top_blob.diff[0, act_id, row, col] = 1.0
        #else:
        #    raise Exception('source/target blobs should be shaped as ' \
        #                    'if from a conv/fc layer')
        #self.net.backward(start=top_layer_name, end=bottom_layer_name)

        # backprop, re-focusing on particular features at each step
        layer_path += [top_layer_name, bottom_layer_name]
        inv_feature_paths = [[i] for i in path_ids]
        for layer_i, tup in enumerate(zip(layer_path[:-1], layer_path[1:], inv_feature_paths, blob_path)):
            top_layer, bottom_layer, feature_idxs, top_blob_name = tup
            # filter activations along feature paths
            if layer_i == 0:
                self.set_max_pixel(0, feature_idxs[0], top_blob_name)
            else:
                self.filter_feature(0, feature_idxs[0], top_blob_name)
            self.net.backward(start=top_layer, end=bottom_layer)

        # compute weights between neurons (backward is needed to do deconvolution on the conv layers)
        edge_weights = bottom_blob.data[0] * bottom_blob.diff[0]
        if len(edge_weights.shape) == 3:
            edge_weights = edge_weights.mean(axis=(1, 2))
        assert len(edge_weights.shape) == 1
        return abs(edge_weights)



    def tree(self, top_layer_id, act_id):
        '''
        Return a JSON serializable dictionary representing a hierarchy of
        features with root given by the top layer and activation id.
        '''
        top_layer_name = self.config['layers'][top_layer_id]['layer_name']
        top_blob_name = self.config['layers'][top_layer_id]['blob_name']
        bottom_layer_id = self.config['layers']['prev_layer_id']
        bottom_layer_name = self.config['layers'][bottom_layer_id]['layer_name']
        bottom_blob_name = self.config['layers'][bottom_layer_id]['blob_name']

        root = self._expand(top_layer_name, top_blob_name,
                           bottom_layer_name, bottom_blob_name,
                           act_id)
        # TODO: don't allow arbitrary depth tree to be returned
        successor_tree = nx.ego_graph(self.dag, root, radius=len(self.dag))
        tree_dict = json_graph.tree_data(successor_tree, root, attrs={'children': 'children', 'id': 'name'})
        return tree_dict



    # TODO??? ... unnecessary
    def node_from_path(self, path):
        '''
        Expand the tree to include this path and return info of the last node in the path.
        '''
        for info in path:
            blob_name, act_id = info['blob_name'], info['act_id']
            self._info_to_key

        if node_id in tree:
            return tree[node_id]

        
            return self._tracce



        act_id = path['act_id']
        top_layer_name = self.config['layers'][top_layer_id]['layer_name']
        top_blob_name = self.config['layers'][top_layer_id]['blob_name']
        bottom_layer_id = self.config['layers']['prev_layer_id']
        bottom_layer_name = self.config['layers'][bottom_layer_id]['layer_name']
        bottom_blob_name = self.config['layers'][bottom_layer_id]['blob_name']

        root, expanded_nodes = self._expand(top_layer_name, top_blob_name,
                     bottom_layer_name, bottom_blob_name, act_id, num_children, return_expanded=True)
        expanded_data = [self.dag.node[node] for node in expanded_nodes]
        return expanded_data


    def expand_path(self, top_layer_name, top_blob_name,
                bottom_layer_name, bottom_blob_name, act_id,
                num_children=5, return_expanded=False):
        '''
        Compute weights between layers w.r.t. a particular activation.

        Given a pair of layers and an activation to focus on in the
        top layer, compute the edge weights between the activation
        and all activations of the previous layer.
        '''
        dag = self.dag
        top_node = self._node_name([top_blob_name], [act_id])
        img_blob = self.net.blobs[self.image_blob]
        expanded_nodes = []

        if top_node not in dag:
            dag.add_node(top_node)
            dag.node[top_node]['blob_name'] = top_blob_name
            dag.node[top_node]['act_id'] = act_id
            dag.node[top_node]['path'] = []
            dag.node[top_node]['path_ids'] = []

        # compute weights
        edge_weights = self._compute_weights(top_layer_name, top_blob_name,
                bottom_layer_name, bottom_blob_name, act_id, num_children)

        # reconstruct strong connections
        important_bottom_idxs = edge_weights.argsort()[::-1][:num_children]
        recons = self.reconstruction([bottom_layer_name], [[i] for i in important_bottom_idxs])

        # fill in the dag with edges from top to bottom and meta data
        for k in range(num_children):
            bottom_idx = important_bottom_idxs[k]
            bottom_node = self._node_name([bottom_blob_name], [bottom_idx])

            dag.add_edge(top_node, bottom_node, attr_dict={
                'weight': edge_weights[bottom_idx],
            })
            dag.node[bottom_node]['blob_name'] = bottom_blob_name
            dag.node[bottom_node]['act_id'] = bottom_idx
            dag.node[bottom_node]['path'] = dag.node[top_node]['path'] + [top_layer_name]
            dag.node[bottom_node]['path_ids'] = dag.node[top_node]['path_ids'] + [act_id]

            expanded_nodes.append(bottom_node)

        # return the node which was expanded
        if return_expanded:
            return top_node, expanded_nodes
        else:
            return top_node






# TODO: refactor
def build_max_act_db(blob_name, config, k=5):
    # don't use self.net, which a deploy net (data comes from python)

    def _load_param(with_data=False):
        if with_data:
            spec_fname = config.spec_wdata
        else:
            spec_fname = config.spec_nodata
        net_param = cpb.NetParameter()
        with open(spec_fname, 'r') as f:
            text_format.Merge(f.read(), net_param)
        return net_param

    def _load_net(net_param):
        '''
        Takes a network spec file and returns a NamedTemporaryFile which
        contains the modified spec with ReLUs appropriate for visualization.
        '''
        # TODO: also accept file objects instead of just names?
        net_param = _convert_relus(net_param, relu_type=config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        return caffe.Net(tmpspec.name, config.model_param, caffe.TEST)

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

    def _get_key(blob_name, feature_idx):
        return '{}_{}'.format(blob_name, feature_idx)

    def _showable(img, rescale=False):
        # TODO: don't always assume images in the net are BGR
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

    mean = load_mean_image(config.mean_fname)
    data_net_param = _load_param(with_data=True)
    net = _load_net(data_net_param)
    img_layer_name = _get_blob_layer(data_net_param, config.img_blob_name)
    batch_size = config.batch_size
    n_batches = int(math.ceil(config.num_examples / float(batch_size)))
    layers = {l.name: l for l in data_net_param.layer}
    dbname = layers[config.data_layer_name].data_param.source
    n_db_examples = _len_lmdb(dbname)
    assert n_db_examples == config.num_examples
    assert n_db_examples % batch_size == 0
    example_offset = 0
    # TODO: load the top_k lists from the db and append to them if they already exist
    maxes = defaultdict(lambda: [{'activation': -np.inf} for _ in range(k)])
    logger.info('blob {}'.format(blob_name))
    img_blob = net.blobs[config.img_blob_name]
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
                key = _get_key(blob_name, chan)
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
        key = _get_key(blob_name, chan)
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
                                  act_mult=config.blob_multipliers[blob_name])
            for blob_idx, entry in zip(blob_idxs, blob_entries):
                num = entry['num']
                reconstruction = img_blob.diff[num, :, :, :]
                bbox = _to_bbox(reconstruction, blob_idx)
                entry['reconstruction'] = _showable(reconstruction, rescale=True)
                entry['patch_bbox'] = bbox

        example_offset += batch_size


    logger.info('finished computing maximum activations... writing to db')
    act_env = lmdb.open(config.max_activation_dbname, map_size=recon.config.config.lmdb_map_size)
    with act_env.begin(write=True) as txn:
        for key, top_k in maxes.iteritems():
            s = pkl.dumps(top_k)
            txn.put(key, s)


