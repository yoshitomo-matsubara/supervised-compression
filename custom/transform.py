from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchdistill.datasets.transform import register_transform_class
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import to_tensor, pad


@register_transform_class
class JpegCenterCrop(CenterCrop):
    def __init__(self, size, jpeg_quality=None):
        super().__init__(size)
        self.jpeg_quality = jpeg_quality

    def __call__(self, img):
        img = super().forward(img)
        if self.jpeg_quality is not None:
            img_buffer = BytesIO()
            img.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
            img = Image.open(img_buffer)
        return img


@register_transform_class
class CustomJpegToTensor(object):
    def __init__(self, jpeg_quality=None):
        self.jpeg_quality = jpeg_quality

    def __call__(self, image, target):
        if self.jpeg_quality is not None:
            img_buffer = BytesIO()
            image.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
            image = Image.open(img_buffer)

        image = to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


@register_transform_class
class AdaptivePad(nn.Module):
    def __init__(self, fill=0, padding_mode='constant', factor=128):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode
        self.factor = factor

    def forward(self, x):
        height, width = x.shape[-2:]
        vertical_pad_size = 0 if height % self.factor == 0 else int((height // self.factor + 1) * self.factor - height)
        horizontal_pad_size = 0 if width % self.factor == 0 else int((width // self.factor + 1) * self.factor - width)
        padded_vertical_size = vertical_pad_size + height
        padded_horizontal_size = horizontal_pad_size + width
        assert padded_vertical_size % self.factor == 0 and padded_horizontal_size % self.factor == 0, \
            'padded vertical and horizontal sizes ({}, {}) should be ' \
            'factor of {}'.format(padded_vertical_size, padded_horizontal_size, self.factor)
        padding = [0, 0, horizontal_pad_size, vertical_pad_size]
        return pad(x, padding, self.fill, self.padding_mode), height, width
