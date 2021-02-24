import torch
from torch import nn
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models

from custom.backbone import get_custom_backbone


def retinanet_mobilenet_v3_large_fpn(pretrained=False, progress=True, num_classes=91, pretrained_backbone=True,
                                     trainable_backbone_layers=None, **kwargs):
    # TODO: should be refactored once torchvision 0.8.3 comes out
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
def keypointrcnn_mobilenet_v3_large_fpn(pretrained=False, progress=True, num_classes=2, num_keypoints=17,
                                        rpn_score_thresh=0.05, pretrained_backbone=True,
                                        trainable_backbone_layers=None, **kwargs):
    # TODO: should be refactored once torchvision 0.8.3 comes out
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
