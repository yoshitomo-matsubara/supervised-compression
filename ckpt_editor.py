import argparse
import os

import torch
from torchdistill.common import file_util
from torchvision import models


def get_argparser():
    parser = argparse.ArgumentParser(description='Checkpoint editor')
    parser.add_argument('--source', required=True, help='source checkpoint to be copied')
    parser.add_argument('--target', required=True, help='target checkpoint to be overwritten')
    parser.add_argument('--mode', required=True, choices=['overwrite'], help='edit mode')
    parser.add_argument('--prefix', default='', help='module path prefix')
    parser.add_argument('--output', required=True, help='output file path')
    return parser


def load_ckpt(file_path_or_model_name):
    if os.path.isfile(file_path_or_model_name):
        ckpt = torch.load(file_path_or_model_name, map_location=torch.device('cpu'))
        return ckpt

    if file_path_or_model_name in models.__dict__:
        model = models.__dict__[file_path_or_model_name](pretrained=True)
    elif file_path_or_model_name in models.detection.__dict__:
        model = models.detection.__dict__[file_path_or_model_name](pretrained=True)
    elif file_path_or_model_name in models.segmentation.__dict__:
        model = models.segmentation.__dict__[file_path_or_model_name](pretrained=True)
    else:
        ValueError('model_name `{}` is not expected'.format(file_path_or_model_name))
    return {'model': model.state_dict()}


def replace_parameters(source_ckpt, prefix, target_ckpt):
    source_model_ckpt = source_ckpt['model']
    target_model_ckpt = target_ckpt['model']
    for source_module_path, source_param in source_model_ckpt.items():
        new_module_path = prefix + source_module_path
        if source_module_path in target_model_ckpt and \
                (source_module_path.startswith(prefix) or len(prefix) == 0):
            print('Parameters of `{}` are replaced with those of `{}`'.format(source_module_path, source_module_path))
            target_model_ckpt[source_module_path] = source_param
        elif new_module_path in target_model_ckpt:
            print('Parameters of `{}` are replaced with those of `{}`'.format(new_module_path, source_module_path))
            target_model_ckpt[new_module_path] = source_param


def main(args):
    source = args.source
    target = args.target
    prefix = args.prefix
    source_ckpt = load_ckpt(source)
    target_ckpt = load_ckpt(target)
    edit_mode = args.mode
    if edit_mode == 'overwrite':
        replace_parameters(source_ckpt, prefix, target_ckpt)
    else:
        ValueError('mode `{}` is not expected'.format(edit_mode))

    output_file_path = args.output
    file_util.make_parent_dirs(output_file_path)
    torch.save(target_ckpt, output_file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
