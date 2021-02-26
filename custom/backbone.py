from compressai.layers import GDN1
from compressai.models import CompressionModel
from torch import nn
from torchdistill.common.file_util import get_binary_object_size
from torchvision import models

CUSTOM_BACKBONE_CLASS_DICT = dict()
CUSTOM_LAYER_CLASS_DICT = dict()


def register_custom_backbone_class(cls):
    CUSTOM_BACKBONE_CLASS_DICT[cls.__name__] = cls
    return cls


def register_custom_layer_class(cls):
    CUSTOM_LAYER_CLASS_DICT[cls.__name__] = cls
    return cls


class BaseCustomBottleneckModel(CompressionModel):
    def __init__(self, entropy_bottleneck_channels, analysis_config=None):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)
        self.analysis_config = analysis_config if analysis_config is not None else dict()
        self.file_size_list = list()

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        if self.analysis_config.get('mean_std_file_size', False):
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)


@register_custom_layer_class
class BottleneckResNetLayerWithIGDN(BaseCustomBottleneckModel):
    def __init__(self, num_enc_channels=16, num_target_channels=256, analysis_config=None):
        super().__init__(entropy_bottleneck_channels=num_enc_channels, analysis_config=analysis_config)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 4),
            nn.Conv2d(num_enc_channels * 4, num_enc_channels * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 2),
            nn.Conv2d(num_enc_channels * 2, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )

    def encode(self, x, **kwargs):
        latent = self.encoder(x)
        compressed_latent = self.entropy_bottleneck.compress(latent)
        return compressed_latent, latent.size()[2:]

    def decode(self, compressed_obj, **kwargs):
        compressed_latent, latent_shape = compressed_obj
        latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
        return self.decoder(latent_hat)

    def forward(self, x):
        if not self.training:
            encoded_obj = self.encode(x)
            self.analyze_compressed_object(encoded_obj)
            decoded_obj = self.decode(encoded_obj)
            return decoded_obj

        encoded_obj = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(encoded_obj)
        decoded_obj = self.decoder(y_hat)
        return decoded_obj


@register_custom_layer_class
class BottleneckResNetLayer(BaseCustomBottleneckModel):
    def __init__(self, num_enc_channels=16, num_target_channels=256, analysis_config=None):
        super().__init__(entropy_bottleneck_channels=num_enc_channels, analysis_config=analysis_config)
        # num_first_output_channels = num_enc_channels * 4 # ver. 0
        # num_second_output_channels = num_enc_channels * 2 # ver. 0
        # num_first_output_channels = 64 # ver. 1
        # num_second_output_channels = 64 # ver. 1
        num_first_output_channels = 96
        num_second_output_channels = 64
        num_third_output_channels = num_enc_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_first_output_channels, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_first_output_channels),
            nn.Conv2d(num_first_output_channels, num_second_output_channels,
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_second_output_channels),
            nn.Conv2d(num_second_output_channels, num_third_output_channels,
                      kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_third_output_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_target_channels * 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_target_channels), nn.ReLU(inplace=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )

    def encode(self, x, **kwargs):
        latent = self.encoder(x)
        compressed_latent = self.entropy_bottleneck.compress(latent)
        return compressed_latent, latent.size()[2:]

    def decode(self, compressed_obj, **kwargs):
        compressed_latent, latent_shape = compressed_obj
        latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
        return self.decoder(latent_hat)

    def forward(self, x):
        if not self.training:
            encoded_obj = self.encode(x)
            self.analyze_compressed_object(encoded_obj)
            decoded_obj = self.decode(encoded_obj)
            return decoded_obj

        encoded_obj = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(encoded_obj)
        decoded_obj = self.decoder(y_hat)
        return decoded_obj


@register_custom_backbone_class
class BottleneckResNetBackbone(nn.Module):
    def __init__(self, bottleneck_config, resnet_model=None, resnet_name='resnet50', **resnet_kwargs):
        super().__init__()
        self.bottleneck_layer = get_custom_layer(bottleneck_config['name'], **bottleneck_config['params'])
        if resnet_model is None:
            resnet_model = models.__dict__[resnet_name](**resnet_kwargs)

        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.updated = False

    def forward(self, x):
        z = self.bottleneck_layer(x)
        z = self.layer2(z)
        z = self.layer3(z)
        return self.layer4(z)

    def update(self):
        self.bottleneck_layer.update()
        self.updated = True


def get_custom_layer(layer_name, **kwargs):
    if layer_name not in CUSTOM_LAYER_CLASS_DICT:
        raise ValueError('layer_name `{}` is not expected'.format(layer_name))
    return CUSTOM_LAYER_CLASS_DICT[layer_name](**kwargs)


def get_custom_backbone(backbone_name, **kwargs):
    if backbone_name not in CUSTOM_BACKBONE_CLASS_DICT:
        raise ValueError('backbone_name `{}` is not expected'.format(backbone_name))
    return CUSTOM_BACKBONE_CLASS_DICT[backbone_name](**kwargs)

