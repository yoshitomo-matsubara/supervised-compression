import argparse
import os
from io import BytesIO

import numpy as np
from torchdistill.datasets.transform import CustomCompose, CustomRandomResize
from torchdistill.datasets.util import load_coco_dataset, build_transform
from torchvision.datasets import ImageFolder, VOCSegmentation
from torchvision.transforms import transforms


def get_argparser():
    parser = argparse.ArgumentParser(description='JPEG file size for ImageNet and COCO segmentation datasets')
    parser.add_argument('--dataset', required=True, choices=['imagenet', 'coco_segment', 'pascal_segment'],
                        help='ckpt dir path')
    return parser


def compute_jpeg_file_size_with_transform(dataset, transform, quality):
    file_size_list = list()
    for img in dataset:
        img = transform(img[0])
        img_buffer = BytesIO()
        img.save(img_buffer, 'JPEG', quality=quality)
        file_size_list.append(img_buffer.tell() / 1024)
    file_sizes = np.array(file_size_list)
    print('JPEG quality: {}, File size [KB]: {} ± {}'.format(quality, file_sizes.mean(), file_sizes.std()))


def compute_jpeg_file_size_for_imagenet_dataset():
    dataset = ImageFolder(root=os.path.expanduser('~/dataset/ilsvrc2012/val'))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    compute_jpeg_file_size_with_transform(dataset, transform, 100)
    compute_jpeg_file_size_with_transform(dataset, transform, 90)
    compute_jpeg_file_size_with_transform(dataset, transform, 80)
    compute_jpeg_file_size_with_transform(dataset, transform, 70)
    compute_jpeg_file_size_with_transform(dataset, transform, 60)
    compute_jpeg_file_size_with_transform(dataset, transform, 50)
    compute_jpeg_file_size_with_transform(dataset, transform, 40)
    compute_jpeg_file_size_with_transform(dataset, transform, 30)
    compute_jpeg_file_size_with_transform(dataset, transform, 20)
    compute_jpeg_file_size_with_transform(dataset, transform, 10)


def compute_jpeg_file_size(dataset, quality):
    file_size_list = list()
    for img in dataset:
        img = img[0]
        img_buffer = BytesIO()
        img.save(img_buffer, 'JPEG', quality=quality)
        file_size_list.append(img_buffer.tell() / 1024)
    file_sizes = np.array(file_size_list)
    print('JPEG quality: {}, File size [KB]: {} ± {}'.format(quality, file_sizes.mean(), file_sizes.std()))


def compute_jpeg_file_size_for_cocosegment_dataset():
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
                                is_segment, transforms, split_config.get('jpeg_quality', None))

    compute_jpeg_file_size(dataset, 100)
    compute_jpeg_file_size(dataset, 90)
    compute_jpeg_file_size(dataset, 80)
    compute_jpeg_file_size(dataset, 70)
    compute_jpeg_file_size(dataset, 60)
    compute_jpeg_file_size(dataset, 50)
    compute_jpeg_file_size(dataset, 40)
    compute_jpeg_file_size(dataset, 30)
    compute_jpeg_file_size(dataset, 20)
    compute_jpeg_file_size(dataset, 10)


def compute_jpeg_file_size_with_transform_and_target(dataset, transform, quality):
    file_size_list = list()
    for img in dataset:
        img, _ = transform(img[0], img[1])
        img_buffer = BytesIO()
        img.save(img_buffer, 'JPEG', quality=quality)
        file_size_list.append(img_buffer.tell() / 1024)
    file_sizes = np.array(file_size_list)
    print('JPEG quality: {}, File size [KB]: {} ± {}'.format(quality, file_sizes.mean(), file_sizes.std()))


def compute_jpeg_file_size_for_pascalsegment_dataset():
    dataset = VOCSegmentation(root=os.path.expanduser('~/dataset/'), image_set='val', year='2012')
    transform = CustomCompose([
        CustomRandomResize(min_size=512, max_size=512)
    ])

    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 100)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 90)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 80)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 70)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 60)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 50)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 40)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 30)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 20)
    compute_jpeg_file_size_with_transform_and_target(dataset, transform, 10)


if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()
    if args.dataset == 'imagenet':
        compute_jpeg_file_size_for_imagenet_dataset()
    elif args.dataset == 'coco_segment':
        compute_jpeg_file_size_for_cocosegment_dataset()
    else:
        compute_jpeg_file_size_for_pascalsegment_dataset()
