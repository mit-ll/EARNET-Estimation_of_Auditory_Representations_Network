import numpy as np

default_params = {
    'allow_train' : True,
    'remove_files' : True,
    'trainfile_repeats' : 20, #???
    'fs' : 16000,
    'output_resample' : [1,1],
    'frame_rate' : 16000,
    'win_dur' : 0.040,
    'fb_nfilts' : 480,
    'do_instance_norm' : False,
    'channel_depth' : 120,
    'out_channels' : 80,
    'hi_fs' : [250,500,1000,2000,3000,4000,6000,8000],
    'desc_layers' : 1,
    'ihc_layers' : 8,
    'ngram_layers' : 2,
    'cnn_dim' : 7,
    'cnn_dilation_max' : 0,
    'do_compression' : True,
    'full_dur' : 1.00,
    'loss_type' : 'sdr_ch',
    'loss_ihc_const' : 2e0,
    'loss_ngram_const' : 1e-3,
    'lr' : 1e-4,
    'lr_final' : 1e-6,
    'lr_warmup' : 100,
    'num_hours' : 10000,
    'batch_size' : 16,
    'batch_files' : 4}


def get_emulator_params(model_version=1):
    params = default_params
    if model_version == 1:
        params['channel_depth'] = 160
    elif model_version == 2:
        params['channel_depth'] = 80
    else:
        assert(0)
    return params
