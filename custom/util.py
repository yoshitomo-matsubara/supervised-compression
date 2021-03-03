from collections import OrderedDict

import torch
from torchdistill.common import file_util, module_util
from torchdistill.common.constant import def_logger


logger = def_logger.getChild(__name__)


def check_if_module_exits(module, module_path):
    module_names = module_path.split('.')
    child_module_name = module_names[0]
    if len(module_names) == 1:
        return hasattr(module, child_module_name)

    if not hasattr(module, child_module_name):
        return False
    return check_if_module_exits(getattr(module, child_module_name), '.'.join(module_names[1:]))


def load_bottleneck_model_ckpt(model, ckpt_file_path):
    from custom.model import BottleneckResNet
    if not file_util.check_if_exists(ckpt_file_path):
        return False

    ckpt = torch.load(ckpt_file_path, map_location='cpu')
    if isinstance(model, BottleneckResNet):
        # For classifier
        logger.info('Loading entropy bottleneck parameters for classifier')
        model_ckpt = ckpt['model']
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(model_ckpt.keys()):
            if key.startswith('backbone.bottleneck_layer.'):
                entropy_bottleneck_state_dict[key.replace('backbone.bottleneck_layer.', '')] = model_ckpt.pop(key)

        model.load_state_dict(model_ckpt, strict=False)
        model.backbone.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)
        return True
    elif check_if_module_exits(model, 'backbone.body.bottleneck_layer'):
        # For detector
        logger.info('Loading entropy bottleneck parameters for detector')
        model_ckpt = ckpt['model']
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(model_ckpt.keys()):
            if key.startswith('backbone.body.bottleneck_layer.'):
                entropy_bottleneck_state_dict[key.replace('backbone.body.bottleneck_layer.', '')] = model_ckpt.pop(key)

        model.load_state_dict(model_ckpt, strict=False)
        model.backbone.body.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)
    elif check_if_module_exits(model, 'backbone.bottleneck_layer'):
        # For segmenter
        logger.info('Loading entropy bottleneck parameters for segmenter')
        model_ckpt = ckpt['model']
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(model_ckpt.keys()):
            if key.startswith('backbone.bottleneck_layer.'):
                entropy_bottleneck_state_dict[key.replace('backbone.bottleneck_layer.', '')] = model_ckpt.pop(key)

        model.load_state_dict(model_ckpt, strict=False)
        model.backbone.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)
    return False


def extract_entropy_bottleneck_module(model):
    model_wo_ddp = model.module if module_util.check_if_wrapped(model) else model
    if hasattr(model_wo_ddp, 'bottleneck'):
        entropy_bottleneck_module = module_util.get_module(model_wo_ddp, 'bottleneck.compressor')
        return entropy_bottleneck_module
    elif hasattr(model_wo_ddp, 'backbone') and hasattr(model_wo_ddp.backbone, 'bottleneck_layer'):
        entropy_bottleneck_module = module_util.get_module(model_wo_ddp, 'backbone.bottleneck_layer')
        return entropy_bottleneck_module
    return None
