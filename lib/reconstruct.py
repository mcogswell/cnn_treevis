import os
import os.path as pth
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import google.protobuf.text_format as text_format

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

import config

def reconstruct(spec, model_params, layer_name, dirname, relu_type='DECONV'):
    # load net spec and figure out which blob to use
    net_param = cpb.NetParameter()
    with open(spec, 'r') as f:
        text_format.Merge(f.read(), net_param)
    layers = dict(zip([l.name for l in net_param.layer], net_param.layer))
    layer = layers[layer_name]
    # don't support old net versions for now
    assert len(net_param.layers) == 0
    # for now, only accept layers with one top blob so I don't have to give
    # the blob's name
    assert len(layer.top) == 1
    blob_name = layer.top[0]

    # set relu backprop type and generate temp net
    backprop_types = dict(zip(cpb.ReLUParameter.BackpropType.keys(),
                              cpb.ReLUParameter.BackpropType.values()))
    for layer in net_param.layer:
        if layer.type == 'ReLU':
            layer.relu_param.backprop_type = backprop_types[relu_type]
    tmpspec = tempfile.NamedTemporaryFile()
    tmpspec.write(text_format.MessageToString(net_param))
    tmpspec.flush()

    # initialize the net and forward()
    net = caffe.Net(tmpspec.name, model_params, caffe.TEST)
    blobs = net.forward()
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
