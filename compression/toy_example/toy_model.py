import torch
import torch.nn as nn
import torch.optim as optim
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1 as GDN


class ImgModel(CompressionModel):
    """Simple autoencoder with a factorized prior """

    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)

        self.encode = nn.Sequential(
            nn.Conv2d(3, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN(N),
            nn.Conv2d(N, N, 5, 2, 2), 
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, 2, 2, 1)
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods