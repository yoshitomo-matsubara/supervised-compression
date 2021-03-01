import torch
from torch import nn
from torchdistill.models.custom.bottleneck.detection.resnet_backbone import Bottleneck4SmallResNet, \
    Bottleneck4LargeResNet
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.retinanet import RetinaNet, model_urls as retinanet_model_urls
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from custom.backbone import get_custom_backbone


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


def _custom_resnet_fpn_backbone(backbone_name, backbone_params_config, norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                                extra_blocks=None):
    layer1_config = backbone_params_config.get('layer1', None)
    layer1 = None
    if layer1_config is not None:
        compressor_config = layer1_config.get('compressor', None)
        compressor = None if compressor_config is None \
            else get_bottleneck_processor(compressor_config['name'], **compressor_config['params'])
        decompressor_config = layer1_config.get('decompressor', None)
        decompressor = None if decompressor_config is None \
            else get_bottleneck_processor(decompressor_config['name'], **decompressor_config['params'])

        layer1_type = layer1_config['type']
        if layer1_type == 'Bottleneck4SmallResNet' and backbone_name in {'custom_resnet18', 'custom_resnet34'}:
            layer1 = Bottleneck4SmallResNet(layer1_config['bottleneck_channel'], compressor, decompressor)
        elif layer1_type == 'Bottleneck4LargeResNet'\
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], compressor, decompressor)

    prefix = 'custom_'
    start_idx = backbone_name.find(prefix) + len(prefix)
    org_backbone_name = backbone_name[start_idx:] if backbone_name.startswith(prefix) else backbone_name
    backbone = resnet.__dict__[org_backbone_name](
        pretrained=backbone_params_config.get('pretrained', False),
        norm_layer=norm_layer
    )
    if layer1 is not None:
        backbone.layer1 = layer1

    trainable_layers = backbone_params_config.get('trainable_backbone_layers', 3)
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    returned_layers = backbone_params_config.get('returned_layers', [2, 3, 4])
    if returned_layers is None:
        returned_layers = [2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


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

    backbone_model = \
        _custom_resnet_fpn_backbone(backbone_name, backbone_params_config, extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone_model, num_classes, **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = load_state_dict_from_url(retinanet_model_urls['retinanet_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_class
class BottleneckResNet(nn.Module):
    def __init__(self, backbone, resnet_model):
        super().__init__()
        self.backbone = backbone
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc
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
    resnet_model = models.__dict__[base_model_name](**base_model_config)
    backbone = get_custom_backbone(backbone_name, resnet_model=resnet_model, **backbone_config)
    return BottleneckResNet(backbone, resnet_model)
