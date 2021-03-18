import torch
from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import check_if_exists
from torchdistill.models.custom.bottleneck.classification.resnet import Bottleneck4ResNet152, CustomResNet
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN, model_urls as keypointrcnn_model_urls
from torchvision.models.detection.retinanet import RetinaNet, model_urls as retinanet_model_urls
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models.segmentation.segmentation import model_urls as segm_model_urls
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, LastLevelP6P7

from custom.backbone import get_custom_backbone, _custom_resnet_fpn_backbone, custom_resnet_backbone4seg
from custom.util import load_bottleneck_model_ckpt

logger = def_logger.getChild(__name__)


def retinanet_mobilenet_v3_large_fpn(pretrained=False, num_classes=91, pretrained_backbone=True,
                                     trainable_backbone_layers=None, **kwargs):
    # TODO: should be refactored once torchvision 0.8.2 > comes out
    from torchvision.models.detection.retinanet import RetinaNet
    from torchvision.models.detection.backbone_utils import _validate_trainable_layers, mobilenet_backbone
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    # check default parameters and by default set it to 3 if possible
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3)

    if pretrained:
        pretrained_backbone = False
    backbone = mobilenet_backbone("mobilenet_v3_large", pretrained_backbone, True,
                                  returned_layers=[4, 5], trainable_layers=trainable_backbone_layers)

    anchor_sizes = ((128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = RetinaNet(backbone, num_classes, anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios), **kwargs)
    return model


@register_model_func
def keypointrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=2, num_keypoints=17,
                                        rpn_score_thresh=0.05, pretrained_backbone=True,
                                        trainable_backbone_layers=None, **kwargs):
    # TODO: should be refactored once torchvision 0.8.2 > comes out
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
    from torchvision.models.detection.backbone_utils import _validate_trainable_layers, mobilenet_backbone
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3)

    if pretrained:
        pretrained_backbone = False
    backbone = mobilenet_backbone("mobilenet_v3_large", pretrained_backbone, True,
                                  trainable_layers=trainable_backbone_layers)

    anchor_sizes = ((32, 64, 128, 256, 512, ), ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = KeypointRCNN(backbone, num_classes, num_keypoints=num_keypoints, rpn_score_thresh=rpn_score_thresh,
                         rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
                         **kwargs)
    return model


@register_model_func
def custom_resnet50(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                    short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4ResNet152(bottleneck_channel, bottleneck_idx, compressor, decompressor)
    org_model = resnet50(**kwargs)
    return CustomResNet(bottleneck, short_module_names, org_model)


@register_model_func
def custom_retinanet_resnet_fpn(backbone, pretrained=True, progress=True,
                                num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_params_config['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_params_config['pretrained'] = False

    backbone_model =\
        _custom_resnet_fpn_backbone(backbone_name, backbone_params_config, extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone_model, num_classes, **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = load_state_dict_from_url(retinanet_model_urls['retinanet_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def custom_deeplabv3_resnet(backbone, pretrained=True, aux=False, progress=True, num_classes=21, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_params_config['pretrained'] = False

    backbone_model = custom_resnet_backbone4seg(backbone_name, aux, backbone_params_config)
    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = DeepLabHead(inplanes, num_classes)

    inplanes = 2048
    classifier = DeepLabHead(inplanes, num_classes)
    model = DeepLabV3(backbone_model, classifier, aux_classifier)
    if pretrained and (backbone_name.endswith('resnet50') or backbone_name.endswith('resnet101')):
        resnet_name = 'resnet50' if backbone_name.endswith('resnet50') else 'resnet101'
        state_dict = \
            load_state_dict_from_url(segm_model_urls['deeplabv3_{}_coco'.format(resnet_name)], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_class
class BottleneckResNet(nn.Module):
    def __init__(self, backbone, resnet_model):
        super().__init__()
        self.backbone = backbone
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc
        self.inplanes = backbone.inplanes
        self.updated = False
        del resnet_model.avgpool
        del resnet_model.fc

    def forward(self, x):
        z = self.backbone(x)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        return self.fc(z)

    def update(self):
        self.backbone.update()
        self.updated = True


@register_model_func
def bottleneck_resnet(backbone_name, backbone_config, base_model_name, base_model_config):
    if base_model_config.get('norm_layer', '') == 'FrozenBatchNorm2d':
        base_model_config['norm_layer'] = misc_nn_ops.FrozenBatchNorm2d

    resnet_model = models.__dict__[base_model_name](**base_model_config)
    backbone = get_custom_backbone(backbone_name, resnet_model=resnet_model, **backbone_config)
    return BottleneckResNet(backbone, resnet_model)


def bottleneck_resnet_fpn_backbone(backbone, trainable_layers=4, returned_layers=None, extra_blocks=None):
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 4
    layers_to_train = ['layer4', 'layer3', 'layer2', 'bottleneck_layer'][:trainable_layers]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}'if k > 1 else 'bottleneck_layer': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


@register_model_func
def retinanet_bottleneck_resnet_fpn(backbone, pretrained=True, progress=True,
                                    num_classes=91, trainable_backbone_layers=4, returned_layers=None, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    assert 0 <= trainable_backbone_layers <= 4
    base_model_name = backbone['base_model_name']
    base_model_config = backbone['base_model_config']
    backbone_resnet = bottleneck_resnet(backbone_name, backbone_params_config, base_model_name, base_model_config)
    backbone_resnet_ckpt_file_path = backbone.get('ckpt', None)
    if check_if_exists(backbone_resnet_ckpt_file_path):
        backbone_resnet_ckpt = torch.load(backbone_resnet_ckpt_file_path, map_location=torch.device('cpu'))
        logger.info('Loading backbone ckpt')
        if not load_bottleneck_model_ckpt(backbone_resnet, backbone_resnet_ckpt_file_path):
            backbone_resnet.load_state_dict(backbone_resnet_ckpt['model'])

    if backbone.get('update', True):
        logger.info('Updating entropy bottleneck in backbone')
        # It's weird, but we need cuda() as sometimes we face "Floating point exception (core dumped)" when update()
        if torch.cuda.is_available():
            backbone_resnet.cuda()
        backbone_resnet.update()

    if returned_layers is None:
        returned_layers = [2, 3, 4]

    backbone_model = \
        bottleneck_resnet_fpn_backbone(backbone_resnet.backbone, trainable_layers=trainable_backbone_layers,
                                       returned_layers=returned_layers, extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone_model, num_classes, **kwargs)
    if pretrained and base_model_name.endswith('resnet50'):
        state_dict = load_state_dict_from_url(retinanet_model_urls['retinanet_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def keypointrcnn_bottleneck_resnet_fpn(backbone, pretrained=True, progress=True, num_classes=91, num_keypoints=17,
                                       trainable_backbone_layers=4, returned_layers=None, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    assert 0 <= trainable_backbone_layers <= 4
    base_model_name = backbone['base_model_name']
    base_model_config = backbone['base_model_config']
    backbone_resnet = bottleneck_resnet(backbone_name, backbone_params_config, base_model_name, base_model_config)
    backbone_resnet_ckpt_file_path = backbone.get('ckpt', None)
    if check_if_exists(backbone_resnet_ckpt_file_path):
        backbone_resnet_ckpt = torch.load(backbone_resnet_ckpt_file_path, map_location=torch.device('cpu'))
        logger.info('Loading backbone ckpt')
        if not load_bottleneck_model_ckpt(backbone_resnet, backbone_resnet_ckpt_file_path):
            backbone_resnet.load_state_dict(backbone_resnet_ckpt['model'])

    if backbone.get('update', True):
        logger.info('Updating entropy bottleneck in backbone')
        # It's weird, but we need cuda() as sometimes we face "Floating point exception (core dumped)" when update()
        if torch.cuda.is_available():
            backbone_resnet.cuda()
        backbone_resnet.update()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    backbone_model = \
        bottleneck_resnet_fpn_backbone(backbone_resnet.backbone,
                                       trainable_layers=trainable_backbone_layers, returned_layers=returned_layers)
    model = KeypointRCNN(backbone_model, num_classes, num_keypoints=num_keypoints, **kwargs)
    if pretrained and base_model_name.endswith('resnet50'):
        state_dict = \
            load_state_dict_from_url(keypointrcnn_model_urls['keypointrcnn_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def bottleneck_segm_model(name, backbone_name, backbone, num_classes, aux, pretrained_backbone=True):
    if 'resnet' in backbone_name:
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
    elif 'mobilenet_v3' in backbone_name:
        # backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, _dilated=True).features
        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    return_layers = {out_layer: 'out'}
    if aux:
        return_layers[aux_layer] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    classifier = model_map[name][0](out_inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def load_segm_model(arch_type, base_model_name, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
        kwargs['pretrained_backbone'] = False

    model = bottleneck_segm_model(arch_type, base_model_name, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + base_model_name + '_coco'
        model_url = segm_model_urls.get(arch, None)
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def deeplabv3_bottleneck_resnet50(backbone, pretrained=True, progress=True, num_classes=21, aux_loss=None, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    base_model_name = backbone['base_model_name']
    base_model_config = backbone['base_model_config']
    backbone_resnet = bottleneck_resnet(backbone_name, backbone_params_config, base_model_name, base_model_config)
    backbone_resnet_ckpt_file_path = backbone.get('ckpt', None)
    if check_if_exists(backbone_resnet_ckpt_file_path):
        backbone_resnet_ckpt = torch.load(backbone_resnet_ckpt_file_path, map_location=torch.device('cpu'))
        logger.info('Loading backbone ckpt')
        if not load_bottleneck_model_ckpt(backbone_resnet, backbone_resnet_ckpt_file_path):
            backbone_resnet.load_state_dict(backbone_resnet_ckpt['model'])

    if backbone.get('update', True):
        logger.info('Updating entropy bottleneck in backbone')
        # It's weird, but we need cuda() as sometimes we face "Floating point exception (core dumped)" when update()
        if torch.cuda.is_available():
            backbone_resnet.cuda()
        backbone_resnet.update()

    return load_segm_model('deeplabv3', base_model_name, backbone_resnet.backbone, pretrained, progress,
                           num_classes, aux_loss, **kwargs)
