from torch import nn
from torchdistill.common.file_util import get_binary_object_size
from torchdistill.datasets.util import build_transform

CUSTOM_MODEL_CLASS_DICT = dict()


def register_custom_model_class(cls):
    CUSTOM_MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


class BaseCustomModel(nn.Module):
    def __init__(self, analysis_config):
        super().__init__()
        self.analysis_config = analysis_config
        self.file_size_list = list()

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        if self.analysis_config.get('mean_std_file_size', False):
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)


@register_custom_model_class
class InputCompressionModel(BaseCustomModel):
    def __init__(self, compressor, classifier, post_transform_params=None, analysis_config=None):
        super().__init__(analysis_config)
        self.compressor = compressor
        self.classifier = classifier
        self.post_transform = build_transform(post_transform_params)
        self.file_size_list = list()

    def forward(self, x):
        compressed_obj = self.compressor.compress(x)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(compressed_obj)

        decompressed_obj = self.compressor.decompress(compressed_obj)
        if self.post_transform is not None:
            decompressed_obj = self.post_transform(decompressed_obj)
        return self.classifier(decompressed_obj)


@register_custom_model_class
class BottleneckInjectedModel(BaseCustomModel):
    def __init__(self, classifier, analysis_config=None, *kwargs):
        super().__init__(analysis_config)
        self.classifier = classifier

    def forward(self, x):
        if not self.training and self.analysis_config is not None:
            z = self.classifier.bottleneck.encoder(x)
            compressed_obj = self.classifier.bottleneck.compressor.compress(z)
            self.analyze_compressed_object(compressed_obj)
        return self.classifier(x)


def get_custom_model(model_name, compressor, classifier, **kwargs):
    if model_name not in CUSTOM_MODEL_CLASS_DICT:
        raise ValueError('model_name `{}` is not expected'.format(model_name))
    return CUSTOM_MODEL_CLASS_DICT[model_name](compressor=compressor, classifier=classifier, **kwargs)
