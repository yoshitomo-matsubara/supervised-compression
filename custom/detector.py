from torch import nn
from torchdistill.common.file_util import get_binary_object_size
from torchvision.models.detection.image_list import ImageList

CUSTOM_DETECTOR_CLASS_DICT = dict()


def register_custom_classifier_class(cls):
    CUSTOM_DETECTOR_CLASS_DICT[cls.__name__] = cls
    return cls


class TransformWrapper(nn.Module):
    def __init__(self, transform, compressor, analysis_config=None):
        super().__init__()
        self.org_transform = transform
        self.compressor = compressor
        self.analysis_config = analysis_config
        self.file_size_list = list()

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        if self.analysis_config.get('mean_std_file_size', False):
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)

    def compress_decompress(self, x):
        compressed_obj = self.compressor.compress(x)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(compressed_obj)
        return self.compressor.decompress(compressed_obj)

    def forward(self, images, targets):
        images, targets = self.org_transform(images, targets)
        if isinstance(images, ImageList):
            images.tensors = self.compress_decompress(images.tensors)
        else:
            images = self.compress_decompress(images)
        return images, targets


@register_custom_classifier_class
class InputCompressionDetector(nn.Module):
    def __init__(self, compressor, detector, analysis_config=None):
        super().__init__()
        self.detector = detector
        self.detector.transform = TransformWrapper(self.detector.transform, compressor, analysis_config)

    def forward(self, *args):
        return self.detector(*args)


def get_custom_model(model_name, compressor, classifier, **kwargs):
    if model_name not in CUSTOM_DETECTOR_CLASS_DICT:
        raise ValueError('model_name `{}` is not expected'.format(model_name))
    return CUSTOM_DETECTOR_CLASS_DICT[model_name](compressor=compressor, classifier=classifier, **kwargs)
