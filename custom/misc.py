from torch import nn
from torchvision.transforms import functional
from torchdistill.datasets.transform import register_transform_class


@register_transform_class
class Crop(nn.Module):
    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, z):
        return functional.crop(z, self.top, self.left, self.height, self.width)


