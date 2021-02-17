import torch.nn as nn
from compressai.layers import GDN1
from compressai.models import CompressionModel, MeanScaleHyperprior
from torchdistill.common.file_util import get_binary_object_size
from torchdistill.models.custom.bottleneck.processor import register_bottleneck_processor
from torchvision.transforms import functional

from compression.registry import register_compression_model_class, register_compression_model_func


class BaseCompressionModel(CompressionModel):
    def __init__(self, entropy_bottleneck_channels=128):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)

    def compress(self, *args, **kwargs):
        raise NotImplementedError

    def decompress(self, *args, **kwargs):
        raise NotImplementedError


@register_compression_model_class
class FactorizedPriorAE(BaseCompressionModel):
    """Simple autoencoder with a factorized prior """
    def __init__(self, entropy_bottleneck_channels=128):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)
        N = entropy_bottleneck_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, 2, 2, 1)
        )

    def compress(self, x, **kwargs):
        latent = self.encoder(x)
        compressed_latent = self.entropy_bottleneck.compress(latent)
        return compressed_latent, latent.size()[2:]

    def decompress(self, compressed_obj, **kwargs):
        compressed_latent, latent_shape = compressed_obj
        latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
        return self.decoder(latent_hat)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return x_hat, y_likelihoods


@register_compression_model_class
class HierarchicalPriorAE(MeanScaleHyperprior):
    """Simple autoencoder with a hierarchical prior """
    def __init__(self, N=128, M=192):
        super().__init__(N, M)

    def forward(self, x):
        output = super().forward(x)
        return output['x_hat'], output['likelihoods']['y'], output['likelihoods']['z']

    def compress(self, x):
        dic = super().compress(x)
        strings = dic['strings']
        shape = dic['shape']
        return strings, shape

    def decompress(self, strings, shape):
        dic = super().decompress(strings, shape)
        return dic['x_hat']


@register_compression_model_class
@register_bottleneck_processor
class MiddleCompressionModel(BaseCompressionModel):
    @staticmethod
    def _build_encoder(num_input_channels, num_mid_channels, num_convs):
        layer_list = list()
        for i in range(num_convs):
            conv_layer = nn.Conv2d(num_input_channels, num_mid_channels, 5, 2, 2)
            num_input_channels = num_mid_channels
            layer_list.append(conv_layer)
            if i < num_convs - 1:
                layer_list.append(GDN1(num_mid_channels))
        return nn.Sequential(*layer_list)

    @staticmethod
    def _build_decoder(num_mid_channels, num_output_channels, num_deconvs):
        layer_list = list()
        for i in range(num_deconvs):
            not_last = i < num_deconvs - 1
            tmp_num_output_channels = num_mid_channels if not_last else num_output_channels
            deconv_layer = nn.ConvTranspose2d(num_mid_channels, tmp_num_output_channels, 5, 2, 2, 1)
            layer_list.append(deconv_layer)
            if not_last:
                layer_list.append(GDN1(num_mid_channels, inverse=True))
        return nn.Sequential(*layer_list)

    """Simple autoencoder with a factorized prior """
    def __init__(self, num_input_channels, entropy_bottleneck_channels, num_convs, pad_kwargs=None, crop_kwargs=None):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)
        self.encoder = self._build_encoder(num_input_channels, entropy_bottleneck_channels, num_convs)
        self.decoder = self._build_decoder(entropy_bottleneck_channels, num_input_channels, num_convs)
        self.pad_kwargs = pad_kwargs
        self.crop_kwargs = crop_kwargs
        self.file_size_list = list()

    def compress(self, x, **kwargs):
        latent = self.encoder(x)
        compressed_latent = self.entropy_bottleneck.compress(latent)
        return compressed_latent, latent.size()[2:]

    def decompress(self, compressed_obj, **kwargs):
        compressed_latent, latent_shape = compressed_obj
        latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
        return self.decoder(latent_hat)

    def forward(self, x):
        if self.pad_kwargs is not None:
            x = functional.pad(x, **self.pad_kwargs)

        if not self.training:
            compressed_obj = self.compress(x)
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)
            x_hat = self.decompress(compressed_obj)
            if self.crop_kwargs is not None:
                x_hat = functional.crop(x_hat, **self.crop_kwargs)
            return x_hat

        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        if self.crop_kwargs is not None:
            x_hat = functional.crop(x_hat, **self.crop_kwargs)
        return x_hat, y_likelihoods


@register_compression_model_func
def toy_example(entropy_bottleneck_channels):
    model = ImageCompressionModel(entropy_bottleneck_channels)
    return model
