import torch
from torch import nn
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNet, model_urls
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from custom.backbone import get_custom_backbone


@register_model_class
class BottleneckResNet(nn.Module):
    def __init__(self, backbone, resnet_model):
        super().__init__()
        self.backbone = backbone
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc
        del resnet_model.avgpool
        del resnet_model.fc

    def forward(self, x):
        z = self.backbone(x)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        return self.fc(z)

    def update(self):
        self.backbone.update()


@register_model_func
def bottleneck_resnet(backbone_name, backbone_config, base_model_name, base_model_config):
    resnet_model = models.__dict__[base_model_name](**base_model_config)
    backbone = get_custom_backbone(backbone_name, resnet_model=resnet_model, **backbone_config)
    return BottleneckResNet(backbone, resnet_model)


@register_model_func
def tmp_retinanet_resnet50_fpn(pretrained=False, progress=True,
                               num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256), trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0
    return model
