from io import BytesIO
from typing import List, Tuple, Dict, Optional

from PIL import Image
from torch import nn, Tensor
from torchdistill.common.file_util import get_binary_object_size
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms.functional import to_pil_image, to_tensor, crop

from custom.transform import AdaptivePad

CUSTOM_DETECTOR_CLASS_DICT = dict()


def register_custom_classifier_class(cls):
    CUSTOM_DETECTOR_CLASS_DICT[cls.__name__] = cls
    return cls


class RCNNTransformWithCompression(GeneralizedRCNNTransform):
    def __init__(self, transform, compressor=None, jpeg_quality=None, webp_quality=None,
                 analysis_config=None, adaptive_pad_config=None):
        super().__init__(transform.min_size, transform.max_size, transform.image_mean, transform.image_std)
        self.compressor = compressor
        self.adaptive_pad = AdaptivePad(**adaptive_pad_config) if isinstance(adaptive_pad_config, dict) else None
        self.jpeg_quality = jpeg_quality
        self.webp_quality = webp_quality
        self.analysis_config = analysis_config
        self.file_size_list = list()

    def analyze_compressed_object(self, compressed_obj):
        # Analyze tensor size / file size, etc
        if self.analysis_config.get('mean_std_file_size', False):
            file_size = get_binary_object_size(compressed_obj)
            self.file_size_list.append(file_size)

    def jpeg_compress(self, org_img):
        pil_img = to_pil_image(org_img, mode='RGB')
        img_buffer = BytesIO()
        pil_img.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(img_buffer)

        pil_img = Image.open(img_buffer)
        return to_tensor(pil_img).to(org_img.device)

    def webp_compress(self, org_img):
        pil_img = to_pil_image(org_img, mode='RGB')
        img_buffer = BytesIO()
        pil_img.save(img_buffer, 'WEBP', quality=self.webp_quality)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(img_buffer)

        pil_img = Image.open(img_buffer)
        return to_tensor(pil_img).to(org_img.device)

    def compress_by_model(self, org_img):
        org_img = org_img.unsqueeze(0)
        padded_img, org_height, org_width = self.adaptive_pad(org_img)
        compressed_obj = self.compressor.compress(padded_img)
        if not self.training and self.analysis_config is not None:
            self.analyze_compressed_object(compressed_obj)

        decompressed_obj = self.compressor.decompress(compressed_obj)
        decompressed_obj = crop(decompressed_obj, 0, 0, org_height, org_width)
        return decompressed_obj.squeeze(0)

    def compress(self, org_img):
        if self.jpeg_quality is not None:
            return self.jpeg_compress(org_img)
        elif self.webp_quality is not None:
            return self.webp_compress(org_img)
        return self.compress_by_model(org_img)

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image, target_index = self.resize(image, target_index)
            shape_before_compression = image.shape
            image = self.compress(image)
            shape_after_compression = image.shape
            assert shape_after_compression == shape_before_compression, \
                'Compression should not change tensor shape {} -> {}'.format(shape_before_compression,
                                                                             shape_after_compression)
            image = self.normalize(image)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


@register_custom_classifier_class
class InputCompressionDetector(nn.Module):
    def __init__(self, compressor, detector, analysis_config=None, **kwargs):
        super().__init__()
        self.detector = detector
        self.detector.transform = RCNNTransformWithCompression(self.detector.transform, compressor=compressor,
                                                               analysis_config=analysis_config, **kwargs)

    def forward(self, *args):
        return self.detector(*args)


def get_custom_model(model_name, compressor, detector, **kwargs):
    if model_name not in CUSTOM_DETECTOR_CLASS_DICT:
        raise ValueError('model_name `{}` is not expected'.format(model_name))
    return CUSTOM_DETECTOR_CLASS_DICT[model_name](compressor=compressor, detector=detector, **kwargs)
