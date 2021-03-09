from torchdistill.common.file_util import get_binary_object_size
from torchdistill.models.custom.bottleneck.processor import register_bottleneck_processor, Quantizer


@register_bottleneck_processor
class QuantizerWrapper(object):
    def __init__(self, num_bits, analysis_config=None):
        self.quantizer = Quantizer(num_bits)
        self.analysis_config = analysis_config
        self.file_size_list = list()

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        if self.analysis_config.get('mean_std_file_size', False):
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)

    def __call__(self, z):
        quantized_z = self.quantizer(z)
        if self.analysis_config is not None:
            self.analyze_compressed_object(quantized_z)
        return quantized_z
