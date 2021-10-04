import argparse
import os

import numpy as np
from torchdistill.datasets.transform import CustomCompose, CustomRandomResize
from torchdistill.datasets.util import load_coco_dataset, build_transform
from torchvision.datasets import ImageFolder, VOCSegmentation
from torchvision.transforms import transforms

from custom.transform import BPG


def get_argparser():
    parser = argparse.ArgumentParser(description='BPG file size for ImageNet and COCO segmentation datasets')
    parser.add_argument('--dataset', required=True, choices=['imagenet', 'coco_segment', 'pascal_segment'],
                        help='ckpt dir path')
    return parser


def compute_bpg_file_size_with_transform(dataset, quality):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    bpg_codec = BPG(bpg_quality=quality, encoder_path='~/manually_installed/libbpg-0.9.8/bpgenc',
                    decoder_path='~/manually_installed/libbpg-0.9.8/bpgdec')
    file_size_list = list()
    for img in dataset:
        img = transform(img[0])
        img, file_size_kbyte = bpg_codec.run(img)
        file_size_list.append(file_size_kbyte)
    file_sizes = np.array(file_size_list)
    print('BPG quality: {}, File size [KB]: {} ± {}'.format(quality, file_sizes.mean(), file_sizes.std()))


def compute_bpg_file_size_for_imagenet_dataset():
    dataset = ImageFolder(root=os.path.expanduser('~/dataset/ilsvrc2012/val'))
    compute_bpg_file_size_with_transform(dataset, 50)
    compute_bpg_file_size_with_transform(dataset, 45)
    compute_bpg_file_size_with_transform(dataset, 40)
    compute_bpg_file_size_with_transform(dataset, 35)
    compute_bpg_file_size_with_transform(dataset, 30)
    compute_bpg_file_size_with_transform(dataset, 25)
    compute_bpg_file_size_with_transform(dataset, 20)
    compute_bpg_file_size_with_transform(dataset, 15)
    compute_bpg_file_size_with_transform(dataset, 10)
    compute_bpg_file_size_with_transform(dataset, 5)
    compute_bpg_file_size_with_transform(dataset, 0)


def compute_bpg_file_size(dataset, quality):
    file_size_list = list()
    bpg_codec = BPG(bpg_quality=quality, encoder_path='~/manually_installed/libbpg-0.9.8/bpgenc',
                    decoder_path='~/manually_installed/libbpg-0.9.8/bpgdec')
    for img in dataset:
        img = img[0]
        img, file_size_kbyte = bpg_codec.run(img)
        file_size_list.append(file_size_kbyte)
    file_sizes = np.array(file_size_list)
    print('BPG quality: {}, File size [KB]: {} ± {}'.format(quality, file_sizes.mean(), file_sizes.std()))


def compute_bpg_file_size_for_cocosegment_dataset():
    split_config = {
        'images': '~/dataset/coco2017/val2017',
        'annotations': '~/dataset/coco2017/annotations/instances_val2017.json',
        'annotated_only': False,
        'is_segment': True,
        'transforms_params': [
            {'type': 'CustomRandomResize', 'params': {'min_size': 520, 'max_size': 520}}
        ]
    }

    is_segment = split_config.get('is_segment', False)
    compose_cls = CustomCompose if is_segment else None
    transforms = build_transform(split_config.get('transforms_params', None), compose_cls=compose_cls)
    dataset = load_coco_dataset(split_config['images'], split_config['annotations'],
                                split_config['annotated_only'], split_config.get('random_horizontal_flip', None),
                                is_segment, transforms, split_config.get('bpg_quality', None))
    compute_bpg_file_size(dataset, 50)
    compute_bpg_file_size(dataset, 45)
    compute_bpg_file_size(dataset, 40)
    compute_bpg_file_size(dataset, 35)
    compute_bpg_file_size(dataset, 30)
    compute_bpg_file_size(dataset, 25)
    compute_bpg_file_size(dataset, 20)
    compute_bpg_file_size(dataset, 15)
    compute_bpg_file_size(dataset, 10)
    compute_bpg_file_size(dataset, 5)
    compute_bpg_file_size(dataset, 0)


def compute_bpg_file_size_with_transform_and_target(dataset, transform, quality):
    bpg_codec = BPG(bpg_quality=quality, encoder_path='~/manually_installed/libbpg-0.9.8/bpgenc',
                    decoder_path='~/manually_installed/libbpg-0.9.8/bpgdec')
    file_size_list = list()
    for img in dataset:
        img, _ = transform(img[0], img[1])
        img, file_size_kbyte = bpg_codec.run(img)
        file_size_list.append(file_size_kbyte)
    file_sizes = np.array(file_size_list)
    print('bpg quality: {}, File size [KB]: {} ± {}'.format(quality, file_sizes.mean(), file_sizes.std()))


def compute_bpg_file_size_for_pascalsegment_dataset():
    dataset = VOCSegmentation(root=os.path.expanduser('~/dataset/'), image_set='val', year='2012')
    transform = CustomCompose([
        CustomRandomResize(min_size=512, max_size=512)
    ])

    compute_bpg_file_size_with_transform_and_target(dataset, transform, 50)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 45)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 40)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 35)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 30)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 25)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 20)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 15)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 10)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 5)
    compute_bpg_file_size_with_transform_and_target(dataset, transform, 0)


if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()
    if args.dataset == 'imagenet':
        compute_bpg_file_size_for_imagenet_dataset()
    elif args.dataset == 'coco_segment':
        compute_bpg_file_size_for_cocosegment_dataset()
    else:
        compute_bpg_file_size_for_pascalsegment_dataset()
