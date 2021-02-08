import torch
from torch import nn
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models

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
