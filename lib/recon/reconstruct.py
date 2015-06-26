import os
import os.path as pth
import tempfile
import cPickle as pkl
import math
from bisect import bisect
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import lmdb

import caffe
import caffe.proto.caffe_pb2 as cpb
import google.protobuf.text_format as text_format

from cogswell import keyboard

import recon.config as config_mod
from recon.config import config
from recon.util import load_mean_image



backprop_types = dict(zip(cpb.ReLUParameter.BackpropType.keys(),
                          cpb.ReLUParameter.BackpropType.values()))

# main api calls

def canonical_image(net_id, blob_name, feature_idx, k):
    net = nets[net_id]
    net.canonical_image(blob_name, feature_idx, k)




class Reconstructor(object):

    @staticmethod
    def _convert_relus(in_param, relu_type='DECONV'):
        '''
        Takes a NetParameter and returns a new NetParameter with ReLUs
        that have the specified backward pass.
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
                layer.relu_param.backprop_type = backprop_types[relu_type]

        return net_param

    def __init__(self, net_id):
        self.net_id = net_id
        self.config = config.nets[self.net_id]
        self.act_env = lmdb.open(self.config.max_activation_dbname)
        self.mean = load_mean_image(self.config.mean_fname)
        self.net_param = self._load_param()

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
        net_param = Reconstructor._convert_relus(net_param)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)

    '''
    def canonical_image(self, blob_name, feature_idx, k):
        with self.act_env.begin() as txn:
            act_key = self.act_key(blob_name, feature_idx)
            activations = json.loads(db.get(act_key))
        img_keys, locations = zip(*activations)
        vis_dict = self.vis(blob_name,
                            feature_idx,
                            img_keys=img_keys,
                            locations=locations,
                            context_patches=True)
        vis_patches = vis_dict['vis_patches']
        context_patches = vis_dict['context_patches']

        self.net.forward(end=layer_name)
        # concatenate into a batch with k images
        # set net input
    '''

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

    def _showable(self, img):
        # TODO: don't always assume images in the net are BGR
        img = img.transpose([1, 2, 0])
        img = (img + self.mean)[:, :, ::-1]
        img = img.clip(0, 255).astype(np.uint8)
        return img

    def _reconstruct_backward(self, net, net_param, blob_name, blob_idx):
        blob = net.blobs[blob_name]
        blob.diff[:] = 0
        blob.diff[blob_idx] = blob.data[blob_idx]
        layer_name = self._get_blob_layer(net_param, blob_name)
        net.backward(start=layer_name, end=self.img_layer_name)

    def build_max_act_db(self, blob_names, k=5):
        # don't use self.net, which a deploy net (data comes from python)
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        self.img_layer_name = self._get_blob_layer(data_net_param, self.config.img_blob_name)
        batch_size = self.config.batch_size
        n_batches = int(math.ceil(self.config.num_examples / float(batch_size)))
        example_offset = 0
        # accumulate this over all batches
        # maxes[key]
        maxes = defaultdict(lambda: [{'activation': -np.inf} for _ in range(k)])

        for batch_i in xrange(n_batches):
            net.forward()
            print('batch {}'.format(batch_i))

            for blob_name in blob_names:
                img_blob = net.blobs[self.config.img_blob_name]
                blob = net.blobs[blob_name]
                assert blob.data.shape[0] == batch_size
                n_features = blob.data.shape[1]
                print('blob {}'.format(blob_name))

                for chan in xrange(n_features): # channel
                    key = self._get_key(blob_name, chan)
                    # highest at the right
                    top_k = maxes[key]
                    flat_idx = blob.data[:, chan, :, :].argmax()
                    num, height, width = np.unravel_index(flat_idx,
                                            (batch_size,)+blob.data.shape[2:])
                    blob_idx = (num, chan, height, width)
                    act = blob.data[blob_idx]

                    if act > top_k[0]['activation']:
                        img = img_blob.data[num, :, :, :]
                        self._reconstruct_backward(net, data_net_param, blob_name, blob_idx)
                        reconstruction = img_blob.diff[num, :, :, :]
                        entry = {
                            'example_idx': example_offset + num,
                            'feature_map_loc': (height, width),
                            'img': self._showable(img),
                            'reconstruction': self._showable(reconstruction),
                            'activation': act,
                        }
                        top_k_idx = bisect([v['activation'] for v in top_k], act)
                        top_k.insert(top_k_idx, entry)
                        top_k = top_k[1:]

            example_offset += batch_size




def reconstruct(spec, model_params, layer_name, dirname, relu_type='DECONV'):
    net = reconstruction_spec(spec, model_params, relu_type=relu_type)

    tmpspec = tempfile.NamedTemporaryFile()
    tmpspec.write(text_format.MessageToString(net_param))
    tmpspec.flush()

    # initialize the net and forward()
    net = caffe.Net(tmpspec.name, model_params, caffe.TEST)
    return net

    # for now, only accept layers with one top blob so I don't have to give
    # the blob's name
    assert len(layer.top) == 1
    blob_name = layer.top[0]

    net.forward()
    blob = net.blobs[blob_name]

    num = blob.data.shape[0]
    max_idxs = []
    for n in xrange(num):
        flat_idx = blob.data[n].argmax()
        idx = np.unravel_index(flat_idx, blob.data.shape[1:])
        blob.diff[n][idx] = blob.data[n][idx]
        max_idxs.append(idx)

    net.backward(start=layer_name, end='conv1')
    mean = config.load_mean_image()[15:-14, 15:-14]
    imgs = net.blobs['data'].data.transpose([0, 2, 3, 1])
    recons = net.blobs['data'].diff.transpose([0, 2, 3, 1])

    def showable(img):
        img = (img + mean)[:, :, ::-1]
        img = img.clip(0, 255).astype(np.uint8)
        return img

    def show(i):
        img = showable(imgs[i])
        # TODO: this needs to be set somehow
        reimg = showable(32*recons[i])
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(reimg)
        plt.subplot(223)
        patchimg = np.copy(img)
        patchimg[recons[i] != 0] = reimg[recons[i] != 0]
        plt.imshow(patchimg)
        plt.title('filter idx: {}'.format(max_idxs[i]))
        try:
            os.mkdir(pth.join(dirname, layer_name))
        except OSError:
            pass
        plt.savefig(pth.join(dirname, layer_name, 'recon_{}_{}_{}.jpg'.format(layer_name, relu_type, i)))

    for i in range(num):
        show(i)

    tmpspec.close()
