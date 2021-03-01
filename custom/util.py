from collections import OrderedDict

import torch
from torchdistill.common import file_util, module_util
from torchdistill.common.constant import def_logger

from custom.model import BottleneckResNet

logger = def_logger.getChild(__name__)


def load_bottleneck_model_ckpt(model, ckpt_file_path):
    if not file_util.check_if_exists(ckpt_file_path):
        return False

    # For classifier
    if isinstance(model, BottleneckResNet):
        logger.info('Loading entropy bottleneck parameters')
        ckpt = torch.load(ckpt_file_path, map_location='cpu')
        model_ckpt = ckpt['model']
        eb_state_dict = OrderedDict()
        for key in list(model_ckpt.keys()):
            if key.startswith('backbone.bottleneck_layer.'):
                eb_state_dict[key.replace('backbone.bottleneck_layer.', '')] = model_ckpt.pop(key)

        model.load_state_dict(model_ckpt, strict=False)
        model.backbone.bottleneck_layer.load_state_dict(eb_state_dict)
        return True
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
