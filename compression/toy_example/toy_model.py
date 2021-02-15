import torch
import torch.nn as nn
import torch.optim as optim
from compressai.models import CompressionModel, MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1 as GDN


class ImgModel(CompressionModel):
    """Simple autoencoder with a factorized prior """
    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
        )

        self.decoder = nn.Sequential(nn.ConvTranspose2d(N, N, 5, 2, 2, 1), GDN(N, inverse=True),
                                    nn.ConvTranspose2d(N, N, 5, 2, 2, 1), GDN(N, inverse=True),
                                    nn.ConvTranspose2d(N, N, 5, 2, 2, 1), GDN(N, inverse=True),
                                    nn.ConvTranspose2d(N, 3, 5, 2, 2, 1))

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return x_hat, y_likelihoods


class HyperImgModel(MeanScaleHyperprior):
    """Simple autoencoder with a hierachical prior """
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