import os
import os.path as pth
import tempfile
import cPickle as pkl
import math
from bisect import bisect
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.util.montage as montage

import lmdb

import caffe
import caffe.proto.caffe_pb2 as cpb
import google.protobuf.text_format as text_format

from cogswell import keyboard

from recon.config import config, relu_backward_types
from recon.util import load_mean_image

import logging
logger = logging.getLogger(config.logger.name)

# main api calls

def canonical_image(net_id, blob_name, feature_idx, k):
    net = nets[net_id]
    net.canonical_image(blob_name, feature_idx, k)




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
        net_param = Reconstructor._convert_relus(net_param, relu_type=self.config.relu_type)

        tmpspec = tempfile.NamedTemporaryFile(delete=False)
        with tmpspec as f:
            tmpspec.write(text_format.MessageToString(net_param))
        tmpspec.close()

        return caffe.Net(tmpspec.name, self.config.model_param, caffe.TEST)

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

    def _to_bbox(self, img, row, col):
        '''
        Take an image which is mostly 0s and return the smallest
        bounding box which contains all non-0 entries.

        img     array of size (c, h, w)
        '''
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
        self.img_layer_name = self._get_blob_layer(data_net_param, 
                                                   self.config.img_blob_name)
        batch_size = self.config.batch_size
        n_batches = int(math.ceil(self.config.num_examples / float(batch_size)))
        example_offset = 0
        # accumulate this over all batches
        # maxes[key]
        # TODO: load the top_k lists from the db and append to them if they already exist
        maxes = defaultdict(lambda: [{'activation': -np.inf} for _ in range(k)])

        for batch_i in xrange(n_batches):
            net.forward()
            logger.info('batch {}'.format(batch_i))

            for blob_name in blob_names:
                img_blob = net.blobs[self.config.img_blob_name]
                blob = net.blobs[blob_name]
                assert blob.data.shape[0] == batch_size
                n_features = blob.data.shape[1]
                logger.info('blob {}'.format(blob_name))

                for num in xrange(batch_size): # examples
                  logger.info('example {}'.format(example_offset + num))
                  for chan in xrange(n_features): # channel
                    key = self._get_key(blob_name, chan)
                    # highest at the right
                    top_k = maxes[key]
                    flat_idx = blob.data[num, chan, :, :].argmax()
                    height, width = np.unravel_index(flat_idx, blob.data.shape[2:])
                    blob_idx = (num, chan, height, width)
                    act = blob.data[blob_idx]
                    assert act == blob.data[num, chan, height, width]

                    if act > top_k[0]['activation']:
                        img = img_blob.data[num, :, :, :]
                        self._reconstruct_backward(net, 
                                                   data_net_param, 
                                                   blob_name, 
                                                   blob_idx)
                        # TODO: instead of mult, use a fixed value, perhaps the 
                        # max over activations in the dataset
                        mult = self.config['blob_multipliers'][blob_name]
                        reconstruction = mult * img_blob.diff[num, :, :, :]
                        bbox = self._to_bbox(reconstruction, height, width)
                        entry = {
                            'example_idx': example_offset + num,
                            'feature_map_loc': (height, width),
                            'img': self._showable(img),
                            'reconstruction': self._showable(reconstruction),
                            'activation': act,
                            'patch_bbox': bbox,
                        }
                        top_k_idx = bisect([v['activation'] for v in top_k], act)
                        top_k.insert(top_k_idx, entry)
                        del top_k[0]

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

    def reconstruct(self, blob_name):
        data_net_param = self._load_param(with_data=True)
        net = self._load_net(data_net_param)
        self.img_layer_name = self._get_blob_layer(data_net_param, 
                                                   self.config.img_blob_name)
        net.forward()
        img_blob = net.blobs[self.config.img_blob_name]
        blob = net.blobs[blob_name]

        num = blob.data.shape[0]
        max_idxs = []
        for n in xrange(num):
            flat_idx = blob.data[n].argmax()
            idx = np.unravel_index(flat_idx, blob.data.shape[1:])
            blob.diff[n][idx] = blob.data[n][idx]
            max_idxs.append(idx)

        layer_name = self._get_blob_layer(data_net_param, blob_name)
        net.backward(start=layer_name, end=self.img_layer_name)

        imgs = img_blob.data
        recons = img_blob.diff

        return imgs, recons
