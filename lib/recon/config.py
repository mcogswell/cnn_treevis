from easydict import EasyDict as edict

config = edict({
# 50 GB
'lmdb_map_size': 50 * (1024**3),
'nets': {
    'caffenet_1000': {
        'max_activation_dbname': 'data/caffenet_1000_max_act_lmdb',
        'spec_wdata': 'specs/caffenet_1000_val.prototxt',
        'spec_nodata': 'specs/caffenet_1000_deploy.prototxt',
        'model_param': 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
        'batch_size': 20,
        'num_examples': 1000,
        'mean_fname': 'caffe/data/ilsvrc12/imagenet_mean.binaryproto',
        'img_blob_name': 'data',
    }
}
})

