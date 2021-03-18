import argparse

import torch
from torchdistill.common import file_util


def get_argparser():
    parser = argparse.ArgumentParser(description='Checkpoint editor')
    parser.add_argument('--ckpt', required=True, help='ckpt dir path')
    return parser


def main(args):
    file_paths = file_util.get_file_path_list(args.ckpt)
    for file_path in file_paths:
        if not file_path.endswith('.pt'):
            print('Skip processing `{}` as it may not be ckpt'.format(file_path))
            continue

        ckpt = torch.load(file_path, map_location=torch.device('cpu'))
        for key in list(ckpt.keys()):
            if key.startswith('encode.'):
                param = ckpt.pop(key)
                ckpt[key.replace('encode.', 'encoder.', 1)] = param
            elif key.startswith('decode.'):
                param = ckpt.pop(key)
                ckpt[key.replace('decode.', 'decoder.', 1)] = param

        torch.save(ckpt, file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
