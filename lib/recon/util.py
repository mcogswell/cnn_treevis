import socket
import sys
import datetime
import os.path as pth
import logging

import numpy as np

import caffe
import caffe.proto.caffe_pb2 as cpb

from recon.config import config

from cogswell import keyboard


def load_ilsvrc12_labels(fname):
    with open(fname, 'r') as f:
        return [line.strip() for line in f]

def load_mean_image(fname):
    bp = cpb.BlobProto()
    with open(fname, 'r') as f:
        bp.ParseFromString(f.read())
    # TODO: don't always assume BGR
    mean = np.array(bp.data).reshape([3, 256, 256]).transpose([1, 2, 0])
    return mean[15:-14, 15:-14]

def setup_logging(name, use_stdout=False):
    fh_args = {
        'name': name,
        'host': socket.gethostname(),
        'time': datetime.datetime.now().isoformat(),
    }
    fh = logging.FileHandler(pth.join(config.logger.dir, '{name}_{host}_{time}'.format(**fh_args)))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s-- %(message)s')
    fh.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    # tasks logger, luigi logger
    loggers = [logging.getLogger(config.logger.name)]
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        if use_stdout:
            logger.addHandler(stdout_handler)
    logger = logging.getLogger(config.logger.name)
    # TODO: log rotation (python should be able to do this)

