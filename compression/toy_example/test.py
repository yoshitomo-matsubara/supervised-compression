import torch
import argparse
import os
import torchvision.transforms.functional as VF
from PIL import Image
from toy_model import ImgModel
from torchvision.utils import save_image


def frame_processing(img, factor=2**4):
    W, H = img.size
    W_pad_size, H_pad_size = 0, 0
    if W % factor != 0:
        W_pad_size = (W % factor) // 2
    if H % factor != 0:
        H_pad_size = (H % factor) // 2
    img = VF.pad(img, (W_pad_size, H_pad_size, W_pad_size, H_pad_size))
    img = VF.to_tensor(img)
    return img.unsqueeze(0), (W_pad_size, H_pad_size)


parser = argparse.ArgumentParser(description='Image Compression Model Test')
parser.add_argument("--image-path", default='test.png', type=str, help="Input Image")
parser.add_argument("--output-path", default='test_reconstruction.png', type=str, help="Compressed Image")
parser.add_argument("--string-path", default='string', type=str, help="Output String")
parser.add_argument("--param-path", default='model_params/toyimg-0.0025.pt', type=str, help="Input Model Parameters")
parser.add_argument("--device", default='cuda', type=str, help="device name used in pytorch (cpu or cuda or cuda:N)")
args = parser.parse_args()

imgmodel = ImgModel(N=128)
params = torch.load(f'{args.param_path}', map_location=lambda storage, loc: storage)
imgmodel.load_state_dict(params)
img = Image.open(f'{args.image_path}')
img, (wp, hp) = frame_processing(img)

img = img.to(args.device)
imgmodel.to(args.device)
imgmodel.update()

latent = imgmodel.encode(img)
strings = imgmodel.entropy_bottleneck.compress(latent)

shape = latent.size()[2:]
latent_hat = imgmodel.entropy_bottleneck.decompress(strings, shape)
img_hat = imgmodel.decode(latent_hat).clamp(0, 1)
save_image(img_hat, f'{args.output_path}')
with open(f'{args.string_path}', 'wb') as f:
    f.write(strings[0])
print('compressed_file(byte):', os.path.getsize(args.string_path))
print('compressed_file(bits):', os.path.getsize(args.string_path) * 8)