from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchdistill.datasets.transform import register_transform_class
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import to_tensor, pad
from compressai.utils.bench.codecs import run_command
from tempfile import mkstemp
import os
import time


class BPG(object):
    """
    Modified https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/bench/codecs.py
    """
    fmt = '.bpg'

    def __init__(self, encoder_path, decoder_path, color_mode='ycbcr', encoder='x265',
                 subsampling_mode='444', bit_depth='8', bpg_quality=50):
        if not isinstance(subsampling_mode, str):
            subsampling_mode = str(subsampling_mode)

        if not isinstance(bit_depth, str):
            bit_depth = str(bit_depth)

        if color_mode not in ['ycbcr', 'rgb']:
            raise ValueError(f'Invalid color mode value: `{color_mode}`, which should be either "ycbcr" or "rgb"')

        if encoder not in ['x265', 'jctvc']:
            raise ValueError(f'Invalid encoder value: `{encoder}`, which should be either "x265" or "jctvc"')

        if subsampling_mode not in ['420', '444']:
            raise ValueError(f'Invalid subsampling mode value: `{subsampling_mode}`, which should be either 420 or 444')

        if bit_depth not in ['8', '10']:
            raise ValueError(f'Invalid bit depth value: `{bit_depth}`, which should be either 8 or 10')

        if not 0 <= bpg_quality <= 51:
            raise ValueError(f'Invalid bpg quality value: `{bpg_quality}`, which should be between 0 and 51')

        self.encoder_path = os.path.expanduser(encoder_path)
        self.decoder_path = os.path.expanduser(decoder_path)
        self.color_mode = color_mode
        self.encoder = encoder
        self.subsampling_mode = subsampling_mode
        self.bit_depth = bit_depth
        self.bpg_quality = bpg_quality

    def _get_encode_cmd(self, img_file_path, output_file_path):
        cmd = [
            self.encoder_path,
            '-o',
            output_file_path,
            '-q',
            str(self.bpg_quality),
            '-f',
            self.subsampling_mode,
            '-e',
            self.encoder,
            '-c',
            self.color_mode,
            '-b',
            self.bit_depth,
            img_file_path
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, reconst_file_path):
        cmd = [self.decoder_path, '-o', reconst_file_path, out_filepath]
        return cmd

    def run(self, resized_pil_img):
        fd_i, resized_input_filepath = mkstemp(suffix='.jpg')
        fd_r, reconst_file_path = mkstemp(suffix='.jpg')
        fd_o, output_file_path = mkstemp(suffix=self.fmt)
        resized_pil_img.save(resized_input_filepath, 'JPEG', quality=100)

        # Encode
        start = time.perf_counter()
        run_command(self._get_encode_cmd(resized_input_filepath, output_file_path))
        enc_time = time.perf_counter() - start
        file_size_kbyte = os.stat(output_file_path).st_size / 1024

        # Decode
        start = time.perf_counter()
        run_command(self._get_decode_cmd(output_file_path, reconst_file_path))
        dec_time = time.perf_counter() - start

        # Read image
        reconst_img = Image.open(reconst_file_path).convert('RGB')
        os.close(fd_i)
        os.remove(resized_input_filepath)
        os.close(fd_r)
        os.remove(reconst_file_path)
        os.close(fd_o)
        os.remove(output_file_path)
        return reconst_img, file_size_kbyte, enc_time, dec_time


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
class WebPCenterCrop(CenterCrop):
    def __init__(self, size, webp_quality=None):
        super().__init__(size)
        self.webp_quality = webp_quality

    def __call__(self, img):
        img = super().forward(img)
        if self.webp_quality is not None:
            img_buffer = BytesIO()
            img.save(img_buffer, 'WEBP', quality=self.webp_quality)
            img = Image.open(img_buffer)
        return img


@register_transform_class
class BpgCenterCrop(CenterCrop):
    def __init__(self, size, bpg_config):
        super().__init__(size)
        self.bpg_codec = BPG(**bpg_config)

    def __call__(self, img):
        img = super().forward(img)
        img, file_size_kbyte, enc_time, dec_time = self.bpg_codec.run(img)
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
class CustomWebPToTensor(object):
    def __init__(self, webp_quality=None):
        self.webp_quality = webp_quality

    def __call__(self, image, target):
        if self.webp_quality is not None:
            img_buffer = BytesIO()
            image.save(img_buffer, 'WEBP', quality=self.webp_quality)
            image = Image.open(img_buffer)

        image = to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


@register_transform_class
class CustomBpgToTensor(object):
    def __init__(self, bpg_config):
        self.bpg_codec = BPG(**bpg_config)

    def __call__(self, image, target):
        image, file_size_kbyte, enc_time, dec_time = self.bpg_codec.run(image)
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
