import argparse
import os

import torch
from torchdistill.common import file_util
from torchvision import models


def get_argparser():
    parser = argparse.ArgumentParser(description='Checkpoint editor')
    parser.add_argument('--source', required=True, help='source checkpoint to be copied')
    parser.add_argument('--target', required=True, help='target checkpoint to be overwritten')
    parser.add_argument('--mode', required=True, choices=['overwrite', 'overwrite_w_prefix'], help='edit mode')
    parser.add_argument('--source_prefix', default='', help='source module path prefix')
    parser.add_argument('--target_prefix', default='', help='target module path prefix')
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


def replace_parameter(source_module_path, target_module_path, source_param, target_model_ckpt):
    print('Parameters of `{}` are replaced with those of `{}`'.format(target_module_path,
                                                                      source_module_path))
    target_model_ckpt[target_module_path] = source_param


def overwrite_parameters(source_ckpt, target_prefix, target_ckpt):
    source_model_ckpt = source_ckpt['model']
    target_model_ckpt = target_ckpt['model']
    for source_module_path, source_param in source_model_ckpt.items():
        target_module_path = target_prefix + source_module_path
        if target_module_path in target_model_ckpt:
            replace_parameter(source_module_path, target_module_path, source_param, target_model_ckpt)


def overwrite_parameters_with_prefix(source_ckpt, source_prefix, target_prefix, target_ckpt):
    source_model_ckpt = source_ckpt['model']
    target_model_ckpt = target_ckpt['model']
    for source_module_path, source_param in source_model_ckpt.items():
        new_target_module_path = source_module_path.replace(source_prefix, target_prefix, 1)
        if source_module_path.startswith(source_prefix) and new_target_module_path in target_model_ckpt:
            replace_parameter(source_module_path, new_target_module_path, source_param, target_model_ckpt)


def main(args):
    source = args.source
    target = args.target
    source_prefix = args.source_prefix
    target_prefix = args.target_prefix
    source_ckpt = load_ckpt(source)
    target_ckpt = load_ckpt(target)
    edit_mode = args.mode
    if edit_mode == 'overwrite':
        overwrite_parameters(source_ckpt, target_prefix, target_ckpt)
    elif edit_mode == 'overwrite_w_prefix':
        overwrite_parameters_with_prefix(source_ckpt, source_prefix, target_prefix, target_ckpt)
    else:
        ValueError('mode `{}` is not expected'.format(edit_mode))

    output_file_path = args.output
    file_util.make_parent_dirs(output_file_path)
    torch.save(target_ckpt, output_file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
