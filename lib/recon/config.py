import caffe.proto.caffe_pb2 as cpb

from easydict import EasyDict as edict

relu_backward_types = edict(dict(cpb.ReLUParameter.BackpropType.items()))
relu_backward_types_inv = { v: k for k, v in relu_backward_types.iteritems()}

config = edict({
'logger': {
    'dir': 'logs/',
    'name': 'parvis',
},
# 50 GB
'lmdb_map_size': 50 * (1024**3),
'nets': {
    'caffenet_1000': {
        'relu_type': relu_backward_types.DECONV,
        'spec_wdata': 'specs/caffenet_1000_val.prototxt',
        'spec_nodata': 'specs/caffenet_1000_deploy.prototxt',
        'model_param': 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
        'batch_size': 20,
        'num_examples': 1000,
        'mean_fname': 'caffe/data/ilsvrc12/imagenet_mean.binaryproto',
        'img_blob_name': 'data',
        'blob_multipliers': {
            'conv1': 1.0,
            'conv2': 2.0,
            'conv3': 4.0,
            'conv4': 8.0,
            'conv5': 16.0,
        }
    },
    'caffenet_imnet_val': {
        'relu_type': relu_backward_types.DECONV,
        'spec_wdata': 'specs/caffenet_imnet_val.prototxt',
        'spec_nodata': 'specs/caffenet_deploy.prototxt',
        'model_param': 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
        'batch_size': 20,
        'num_examples': 50000,
        'mean_fname': 'caffe/data/ilsvrc12/imagenet_mean.binaryproto',
        'img_blob_name': 'data',
        'blob_multipliers': {
            'conv1': 1.0,
            'conv2': 2.0,
            'conv3': 4.0,
            'conv4': 8.0,
            'conv5': 16.0,
        }
    }
}
}) 

for net_key, net in config.nets.iteritems():
    relu_type = relu_backward_types_inv[net.relu_type]
    net.max_activation_dbname = \
        'data/{}_max_act_lmdb_{}'.format(net_key, relu_type)
