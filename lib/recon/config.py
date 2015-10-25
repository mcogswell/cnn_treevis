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
        'conv5': 40.0,
        'fc6': 400.0,
        'fc7': 4096.0,
        'fc8': 1000.0,
    },
    # This describes which pairs of caffe layers have (weighted)
    # edges between them for vis purposes. (see Reconstructor.graph)
    # Edges are directed in the same direction as forward prop
    'edges': [
        { # layer end backward at
          'source_layer': 'conv1',
          # layer to start backward from
          'target_layer': 'relu1',
          # blob with source neurons (probably a bottom of source_layer)
          'source_blob': 'data',
          # blob with target neurons
          'target_blob': 'conv1', },
        { 'source_layer': 'conv2',
          'target_layer': 'relu2',
          'source_blob': 'norm1',
          'target_blob': 'conv2', },
        { 'source_layer': 'conv3',
          'target_layer': 'relu3',
          'source_blob': 'norm2',
          'target_blob': 'conv3', },
        { 'source_layer': 'conv4',
          'target_layer': 'relu4',
          'source_blob': 'conv3',
          'target_blob': 'conv4', },
        { 'source_layer': 'conv5',
          'target_layer': 'relu5',
          'source_blob': 'conv4',
          'target_blob': 'conv5', },
        # TODO: add fully connected layers
    ],
    'prev_layer_map': {
        # TODO: make these work
        #'fc7': 'fc6',
        #'fc6': 'conv5',
        'conv5': 'conv4',
        'conv4': 'conv3',
        'conv3': 'conv2',
        'conv2': 'conv1',
        'conv1': 'data',
    },
    'layers': {
        'fc8': { 'blob_name': 'fc8', 'layer_name': 'fc8', },
        'fc7': { 'blob_name': 'fc7', 'layer_name': 'relu7', },
        'fc6': { 'blob_name': 'fc6', 'layer_name': 'relu6', },
        'conv5': { 'blob_name': 'conv5', 'layer_name': 'relu5', },
        'conv4': { 'blob_name': 'conv4', 'layer_name': 'relu4', },
        'conv3': { 'blob_name': 'conv3', 'layer_name': 'relu3', },
        'conv2': { 'blob_name': 'conv2', 'layer_name': 'relu2', },
        'conv1': { 'blob_name': 'conv1', 'layer_name': 'relu1', },
    },
    'layer_to_blob': {
        'fc8': 'fc8',
        'relu7': 'fc7',
        'relu6': 'fc6',
        'relu5': 'conv5',
        'relu4': 'conv4',
        'relu3': 'conv3',
        'relu2': 'conv2',
        'relu1': 'conv1',
    },
    'blob_name_to_layer_name': {
        'fc8': 'fc8',
        'fc7': 'relu7',
        'fc6': 'relu6',
        'conv5': 'relu5',
        'conv4': 'relu4',
        'conv3': 'relu3',
        'conv2': 'relu2',
        'conv1': 'relu1',
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
    'decov_nodrop_val': dict(net_config,**{
        'relu_type': relu_backward_types.GUIDED,
        'spec_wdata': 'specs/alexnet_val.prototxt',
        'spec_nodata': 'specs/alexnet_deploy.prototxt',
        'num_examples': 50000,
        'model_param': 'data/models/alexnet_xcov67_xw.001_nodrop_snap_iter_450000.caffemodel',
    }),
    'nodecov_val': dict(net_config,**{
        'relu_type': relu_backward_types.GUIDED,
        'spec_wdata': 'specs/alexnet_val.prototxt',
        'spec_nodata': 'specs/alexnet_deploy.prototxt',
        'num_examples': 50000,
        'model_param': 'data/models/alexnet_base_snap_iter_450000.caffemodel',
    }),
}
}) 

for net_key, net in config.nets.iteritems():
    relu_type = relu_backward_types_inv[net.relu_type]
    net.max_activation_dbname = \
        'data/{}_max_act_lmdb_{}'.format(net_key, relu_type)
