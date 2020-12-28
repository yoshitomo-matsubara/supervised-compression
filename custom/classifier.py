from torch import nn
from torchdistill.datasets.util import build_transform

CUSTOM_MODEL_CLASS_DICT = dict()


def register_custom_model_class(cls):
    CUSTOM_MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


@register_custom_model_class
class InputCompressionModel(nn.Module):
    def __init__(self, compressor, classifier, post_transform_params=None, analysis_config=None):
        super().__init__()
        self.compressor = compressor
        self.classifier = classifier
        self.post_transform = build_transform(post_transform_params)
        self.analysis_config = analysis_config

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        pass

    def forward(self, x):
        compressed_obj = self.compressor.compress(x)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(compressed_obj)

        decompressed_obj = self.compressor.decompress(compressed_obj)
        if self.post_transform is not None:
            decompressed_obj = self.post_transform(decompressed_obj)
        return self.classifier(decompressed_obj)


def get_custom_model(model_name, compressor, classifier, **kwargs):
    if model_name not in CUSTOM_MODEL_CLASS_DICT:
        raise ValueError('model_name `{}` is not expected'.format(model_name))
    return CUSTOM_MODEL_CLASS_DICT[model_name](compressor, classifier, **kwargs)
