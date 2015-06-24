import numpy as np
import matplotlib.pyplot as plt

import caffe
import caffe.proto.caffe_pb2 as cpb

from cogswell import keyboard

import config

def reconstruct(spec, model_params):
    net = caffe.Net(spec, model_params, caffe.TEST)
    blobs = net.forward()

    blob = net.blobs['conv1']

    num = blob.data.shape[0]
    max_idxs = []
    for n in xrange(num):
        flat_idx = blob.data[n].argmax()
        idx = np.unravel_index(flat_idx, blob.data.shape[1:])
        blob.diff[n][idx] = blob.data[n][idx]

    net.backward(start='relu1', end='conv1')
    mean = config.load_mean_image()[15:-14, 15:-14]
    imgs = net.blobs['data'].data.transpose([0, 2, 3, 1])
    recons = net.blobs['data'].diff.transpose([0, 2, 3, 1])

    def showable(img):
        img = (img + mean)[:, :, ::-1]
        img = img.clip(0, 255).astype(np.uint8)
        return img

    def show(i):
        img = showable(imgs[i])
        reimg = showable(recons[i])
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(reimg)
        plt.subplot(223)
        patchimg = np.copy(img)
        patchimg[recons[i] != 0] = reimg[recons[i] != 0]
        plt.imshow(patchimg)
        plt.show()

    keyboard('hi')
