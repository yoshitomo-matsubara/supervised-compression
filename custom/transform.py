from io import BytesIO

from PIL import Image
from torchdistill.datasets.transform import register_transform_class
from torchvision.transforms import CenterCrop


@register_transform_class
class JpegCenterCrop(CenterCrop):
    def __init__(self, size, jpeg_quality=None):
        super().__init__(size)
        self.jpeg_quality = jpeg_quality

    def __call__(self, img):
        img = super().forward(img)
        if self.jpeg_quality is not None:
            img_buffer = BytesIO()
            img.save(img_buffer, 'JPEG', quality=self.jpeg_quality)
            img = Image.open(img_buffer)
        return img
