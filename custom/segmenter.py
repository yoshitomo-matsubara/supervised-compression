from torch import nn
from torchdistill.common.file_util import get_binary_object_size
from torchdistill.datasets.util import build_transform
from torchvision.transforms.functional import crop

from custom.transform import AdaptivePad

CUSTOM_SEGMENTER_CLASS_DICT = dict()


def register_custom_segmenter_class(cls):
    CUSTOM_SEGMENTER_CLASS_DICT[cls.__name__] = cls
    return cls


class BaseCustomSegmenter(nn.Module):
    def __init__(self, analysis_config):
        super().__init__()
        self.analysis_config = analysis_config
        self.file_size_list = list()

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        if self.analysis_config.get('mean_std_file_size', False):
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)


@register_custom_segmenter_class
class InputCompressionSegmenter(BaseCustomSegmenter):
    def __init__(self, compressor, segmenter, post_transform_params=None,
                 analysis_config=None, adaptive_pad_config=None):
        super().__init__(analysis_config)
        self.compressor = compressor
        self.segmenter = segmenter
        self.adaptive_pad = AdaptivePad(**adaptive_pad_config) if isinstance(adaptive_pad_config, dict) else None
        self.post_transform = build_transform(post_transform_params)
        self.file_size_list = list()

    def forward(self, x):
        org_height, org_width = None, None
        if self.adaptive_pad is not None:
            x, org_height, org_width = self.adaptive_pad(x)

        compressed_obj = self.compressor.compress(x)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(compressed_obj)

        decompressed_obj = self.compressor.decompress(compressed_obj)
        if self.adaptive_pad is not None:
            decompressed_obj = crop(decompressed_obj, 0, 0, org_height, org_width)
        if self.post_transform is not None:
            decompressed_obj = self.post_transform(decompressed_obj)
        return self.segmenter(decompressed_obj)


def get_custom_model(model_name, compressor, segmenter, **kwargs):
    if model_name not in CUSTOM_SEGMENTER_CLASS_DICT:
        raise ValueError('model_name `{}` is not expected'.format(model_name))
    return CUSTOM_SEGMENTER_CLASS_DICT[model_name](compressor=compressor, segmenter=segmenter, **kwargs)
