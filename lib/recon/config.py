import caffe.proto.caffe_pb2 as cpb

from easydict import EasyDict as edict

relu_backward_types = edict(dict(cpb.ReLUParameter.BackpropType.items()))
relu_backward_types_inv = { v: k for k, v in relu_backward_types.iteritems()}

net_config = {
    'model_param': 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
    # NOTE: this is hard coded in the net specs in specs/
    'batch_size': 20,
    'mean_fname': 'caffe/data/ilsvrc12/imagenet_mean.binaryproto',
    'labels_fname': 'caffe/data/ilsvrc12/synset_words.txt',
    'image_blob_name': 'data',
    'prob_blob_name': 'prob',
    'data_layer_name': 'data',
    'blob_multipliers': {
        'conv1': 1.0,
        'conv2': 4.0,
        'conv3': 20.0,
        'conv4': 40.0,
        'conv5': 40.0,
        'fc6': 400.0,
        'fc7': 4096.0,
        'fc8': 1000.0,
    },
    'layers': {
        'fc8':   { 'blob_name': 'fc8',   'layer_name': 'fc8',   'idx': 8, 'prev_layer_id': 'fc7',   'include_in_overview': True },
        'fc7':   { 'blob_name': 'fc7',   'layer_name': 'relu7', 'idx': 7, 'prev_layer_id': 'fc6',   'include_in_overview': True },
        'fc6':   { 'blob_name': 'fc6',   'layer_name': 'relu6', 'idx': 6, 'prev_layer_id': 'conv5', 'include_in_overview': True },
        'conv5': { 'blob_name': 'conv5', 'layer_name': 'relu5', 'idx': 5, 'prev_layer_id': 'conv4', 'include_in_overview': True },
        'conv4': { 'blob_name': 'conv4', 'layer_name': 'relu4', 'idx': 4, 'prev_layer_id': 'conv3', 'include_in_overview': True },
        'conv3': { 'blob_name': 'conv3', 'layer_name': 'relu3', 'idx': 3, 'prev_layer_id': 'conv2', 'include_in_overview': True },
        'conv2': { 'blob_name': 'conv2', 'layer_name': 'relu2', 'idx': 2, 'prev_layer_id': 'conv1', 'include_in_overview': True },
        'conv1': { 'blob_name': 'conv1', 'layer_name': 'relu1', 'idx': 1, 'prev_layer_id': 'data',  'include_in_overview': True },
        'data':  { 'blob_name': 'data',  'layer_name': 'NA',    'idx': 0, 'prev_layer_id': None,    'include_in_overview': False },
    },
}

config = edict({
'logger': {
    'dir': 'logs/',
    'name': 'cnn_treevis',
},
'lmdb_map_size': 1000 * (1024**3),
'nets': {
    # `net_id`s
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
    'caffenet_imnet_val_neg': dict({
        'relu_type': relu_backward_types.GUIDED_NEG,
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
