import caffe.proto.caffe_pb2 as cpb

from easydict import EasyDict as edict

relu_backward_types = edict(dict(cpb.ReLUParameter.BackpropType.items()))
relu_backward_types_inv = { v: k for k, v in relu_backward_types.iteritems()}

net_config = {
    'model_param': 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
    'batch_size': 20,
    'mean_fname': 'caffe/data/ilsvrc12/imagenet_mean.binaryproto',
    'img_blob_name': 'data',
    'data_layer_name': 'data',
    'blob_multipliers': {
        'conv1': 1.0,
        'conv2': 4.0,
        'conv3': 20.0,
        'conv4': 40.0,
        'conv5': 100.0,
        'fc6': 400.0,
        'fc7': 4096.0,
    }
}

config = edict({
'logger': {
    'dir': 'logs/',
    'name': 'parvis',
},
'lmdb_map_size': 1000 * (1024**3),
'nets': {
    'caffenet_1000': dict({
        'relu_type': relu_backward_types.GUIDED,
        'spec_wdata': 'specs/caffenet_1000_val.prototxt',
        'spec_nodata': 'specs/caffenet_1000_deploy.prototxt',
        'num_examples': 1000,
    }, **net_config),
    'caffenet_imnet_val': dict({
        'relu_type': relu_backward_types.GUIDED,
        'spec_wdata': 'specs/caffenet_imnet_val.prototxt',
        'spec_nodata': 'specs/caffenet_deploy.prototxt',
        'num_examples': 50000,
    }, **net_config),
    'caffenet_imnet_train': dict({
        'relu_type': relu_backward_types.GUIDED,
        'spec_wdata': 'specs/caffenet_imnet_train.prototxt',
        'spec_nodata': 'specs/caffenet_deploy.prototxt',
        'num_examples': 1281167,
    }, **net_config),
}
}) 

for net_key, net in config.nets.iteritems():
    relu_type = relu_backward_types_inv[net.relu_type]
    net.max_activation_dbname = \
        'data/{}_max_act_lmdb_{}'.format(net_key, relu_type)
