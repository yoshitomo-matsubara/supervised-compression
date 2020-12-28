import torch.nn as nn
from compressai.layers import GDN1
from compressai.models import CompressionModel

from compression.registry import register_compression_model_class, register_compression_model_func


class BaseCompressionModel(CompressionModel):
    def __init__(self, entropy_bottleneck_channels=128):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)

    def compress(self, *args, **kwargs):
        raise NotImplementedError

    def decompress(self, *args, **kwargs):
        raise NotImplementedError


@register_compression_model_class
class ImageCompressionModel(BaseCompressionModel):
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


@register_compression_model_func
def toy_example(entropy_bottleneck_channels):
    model = ImageCompressionModel(entropy_bottleneck_channels)
    return model
