import os
import numpy as np


class GPUShortageError(Exception):
    pass


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > logs/tmp')
    memory_available = [int(x.split()[2]) for x in open('logs/tmp', 'r').readlines()]
    max_idx = np.argmax(memory_available)
    max_mem = np.max(memory_available)
    if max_mem <= 10240:
        raise GPUShortageError('not enough GPU memory')
    else:
        return max_idx


# training config
device = f'cuda:{get_free_gpu()}'
# device = 'cuda:7'
batch_size = 8
n_step = 1000000
extra_step = 0
scheduler_step = 900000
save_step = 50000
lr = 1e-4
grad_clip = 1.
decay = .1
minf = .1
use_amp = False

# model config
# for ssf warping
ssf_mode = 'scale_space_warp_reparam'
# ssf_mode = 'warp'
base_scale = 1.
# universal
mtype = 'ssf-same'
filter_size = 128
hyper_filter_size = 192
beta = 1.5625e-4

# general
tbdir = 'tblog-img-toy'
# name = 'ssf-af'

data_config = {
    'dataset_name': 'img',
    'data_path': '/srv/disk00/bdl/ruihay1/dataset',
    'sequence_length': 3,
    'img_size': 256,
    'img_channel': 3
}

extra_data_config = {
    'dataset_name': 'vimeo',
    'data_path': '/srv/disk00/bdl/ruihay1/dataset',
    'sequence_length': 3,
    'img_size': 256,
    'img_channel': 3
}
