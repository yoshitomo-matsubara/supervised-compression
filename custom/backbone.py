from compressai.layers import GDN1
from compressai.models import CompressionModel
from torch import nn
from torchdistill.common.file_util import get_binary_object_size
from torchdistill.models.custom.bottleneck.detection.resnet_backbone import Bottleneck4SmallResNet, \
    Bottleneck4LargeResNet
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor
from torchvision import models
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

CUSTOM_BACKBONE_CLASS_DICT = dict()
CUSTOM_LAYER_CLASS_DICT = dict()


def register_custom_backbone_class(cls):
    CUSTOM_BACKBONE_CLASS_DICT[cls.__name__] = cls
    return cls


def register_custom_layer_class(cls):
    CUSTOM_LAYER_CLASS_DICT[cls.__name__] = cls
    return cls


def _custom_resnet_fpn_backbone(backbone_name, backbone_params_config, norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                                extra_blocks=None):
    layer1_config = backbone_params_config.get('layer1', None)
    layer1 = None
    if layer1_config is not None:
        compressor_config = layer1_config.get('compressor', None)
        compressor = None if compressor_config is None \
            else get_bottleneck_processor(compressor_config['name'], **compressor_config['params'])
        decompressor_config = layer1_config.get('decompressor', None)
        decompressor = None if decompressor_config is None \
            else get_bottleneck_processor(decompressor_config['name'], **decompressor_config['params'])

        layer1_type = layer1_config['type']
        if layer1_type == 'Bottleneck4SmallResNet' and backbone_name in {'custom_resnet18', 'custom_resnet34'}:
            layer1 = Bottleneck4SmallResNet(layer1_config['bottleneck_channel'], compressor, decompressor)
        elif layer1_type == 'Bottleneck4LargeResNet'\
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], compressor, decompressor)

    prefix = 'custom_'
    start_idx = backbone_name.find(prefix) + len(prefix)
    org_backbone_name = backbone_name[start_idx:] if backbone_name.startswith(prefix) else backbone_name
    backbone = resnet.__dict__[org_backbone_name](
        pretrained=backbone_params_config.get('pretrained', False),
        norm_layer=norm_layer
    )
    if layer1 is not None:
        backbone.layer1 = layer1

    trainable_layers = backbone_params_config.get('trainable_backbone_layers', 3)
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    returned_layers = backbone_params_config.get('returned_layers', [2, 3, 4])
    if returned_layers is None:
        returned_layers = [2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


def custom_resnet_backbone4seg(backbone_name, uses_aux, backbone_params_config, return_layers=None):
    if return_layers is None:
        return_layers = {'layer4': 'out'}
        if uses_aux:
            return_layers['layer3'] = 'aux'

    layer1_config = backbone_params_config.get('layer1', None)
    layer1 = None
    if layer1_config is not None:
        compressor_config = layer1_config.get('compressor', None)
        compressor = None if compressor_config is None \
            else get_bottleneck_processor(compressor_config['name'], **compressor_config['params'])
        decompressor_config = layer1_config.get('decompressor', None)
        decompressor = None if decompressor_config is None \
            else get_bottleneck_processor(decompressor_config['name'], **decompressor_config['params'])

        layer1_type = layer1_config['type']
        if layer1_type == 'Bottleneck4SmallResNet' and backbone_name in {'custom_resnet18', 'custom_resnet34'}:
            layer1 = Bottleneck4SmallResNet(layer1_config['bottleneck_channel'], compressor, decompressor)
        elif layer1_type == 'Bottleneck4LargeResNet'\
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], compressor, decompressor)

    prefix = 'custom_'
    start_idx = backbone_name.find(prefix) + len(prefix)
    org_backbone_name = backbone_name[start_idx:] if backbone_name.startswith(prefix) else backbone_name
    backbone = resnet.__dict__[org_backbone_name](
        pretrained=backbone_params_config.get('pretrained', True),
        replace_stride_with_dilation=[False, True, True]
    )
    if layer1 is not None:
        backbone.layer1 = layer1
    return IntermediateLayerGetter(backbone, return_layers=return_layers)


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
        self.updated = False

    def update(self, force=False):
        super().update(force=force)
        self.updated = True

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

        # if fine-tuning after "update"
        if self.updated:
            encoded_output = self.encoder(x)
            decoder_input =\
                self.entropy_bottleneck.dequantize(self.entropy_bottleneck.quantize(encoded_output, 'dequantize'))
            decoder_input = decoder_input.detach()
            return self.decoder(decoder_input)

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
        self.updated = False

    def update(self, force=False):
        super().update(force=force)
        self.updated = True

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

        # if fine-tuning after "update"
        if self.updated:
            encoded_output = self.encoder(x)
            decoder_input = \
                self.entropy_bottleneck.dequantize(self.entropy_bottleneck.quantize(encoded_output, 'dequantize'))
            decoder_input = decoder_input.detach()
            return self.decoder(decoder_input)

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
            if resnet_kwargs.pop('norm_layer', '') == 'FrozenBatchNorm2d':
                resnet_model = models.__dict__[resnet_name](norm_layer=misc_nn_ops.FrozenBatchNorm2d, **resnet_kwargs)
            else:
                resnet_model = models.__dict__[resnet_name](**resnet_kwargs)

        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.inplanes = resnet_model.inplanes
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
