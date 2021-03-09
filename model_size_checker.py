import argparse
import os

import torch
from compressai.models import MeanScaleHyperprior
from compressai.zoo.pretrained import load_pretrained
from torchdistill.common import yaml_util
from torchdistill.common.main_util import load_ckpt
from torchdistill.common.module_util import count_params
from torchdistill.models.custom.bottleneck.classification.resnet import CustomResNet
from torchdistill.models.official import OFFICIAL_MODEL_DICT, get_image_classification_model, \
    get_object_detection_model, get_semantic_segmentation_model
from torchdistill.models.registry import get_model

from compression.registry import get_compression_model
from custom.classifier import InputCompressionClassifier, get_custom_model as get_custom_classifier
from custom.detector import InputCompressionDetector, get_custom_model as get_custom_detector
from custom.model import BottleneckResNet
from custom.segmenter import InputCompressionSegmenter, get_custom_model as get_custom_segmenter
from custom.util import check_if_module_exits, load_bottleneck_model_ckpt


def get_argparser():
    parser = argparse.ArgumentParser(description='Check model size')
    parser.add_argument('--classifier', help='classifier config file path')
    parser.add_argument('--detector', help='detector config file path')
    parser.add_argument('--segmenter', help='segmenter config file path')
    parser.add_argument('--model_name', help='model name available in torchvision')
    return parser


def get_model_config(config_file_path):
    config = yaml_util.load_yaml_file(os.path.expanduser(config_file_path))
    models_config = config['models']
    model_config = models_config.get('student_model', None)
    if model_config is None:
        model_config = models_config['model']
    return model_config


def check_model_size(model, model_name):
    if not isinstance(model, (list, tuple)):
        model = [model]

    num_params = sum(count_params(m) for m in model)
    print('{}: {} parameters'.format(model_name, num_params))


def load_classifier(model_config, distributed=False, sync_bn=False):
    if 'compressor' not in model_config:
        model = get_image_classification_model(model_config, distributed, sync_bn)
        if model is None:
            repo_or_dir = model_config.get('repo_or_dir', None)
            model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

        model_ckpt_file_path = model_config['ckpt']
        if not os.path.isfile(model_ckpt_file_path) and 'start_ckpt' in model_config:
            model_ckpt_file_path = model_config['start_ckpt']

        if load_bottleneck_model_ckpt(model, model_ckpt_file_path):
            return model

        load_ckpt(model_ckpt_file_path, model=model, strict=False)
        return model

    # Define compressor
    compressor_config = model_config['compressor']
    compressor = get_compression_model(compressor_config['name'], **compressor_config['params'])
    compressor_ckpt_file_path = compressor_config['ckpt']
    if os.path.isfile(compressor_ckpt_file_path):
        print('Loading compressor parameters')
        state_dict = torch.load(compressor_ckpt_file_path)
        # Old parameter keys do not work with recent version of compressai
        state_dict = load_pretrained(state_dict)
        compressor.load_state_dict(state_dict)

    print('Updating compression model')
    compressor.update()
    # Define classifier
    classifier_config = model_config['classifier']
    classifier = get_image_classification_model(classifier_config, distributed, sync_bn)
    if classifier is None:
        repo_or_dir = classifier_config.get('repo_or_dir', None)
        classifier = get_model(classifier_config['name'], repo_or_dir, **classifier_config['params'])

    classifier_ckpt_file_path = classifier_config['ckpt']
    load_ckpt(classifier_ckpt_file_path, model=classifier, strict=True)
    custom_model = get_custom_classifier(model_config['name'], compressor, classifier, **model_config['params'])
    return custom_model


def check_classifier_size(classifier, model_name):
    if isinstance(classifier, InputCompressionClassifier) and isinstance(classifier.compressor, MeanScaleHyperprior):
        check_model_size([classifier.compressor.g_a, classifier.compressor.h_a, classifier.compressor.h_s],
                         'Encoder in {}'.format(model_name))
        check_model_size([classifier.compressor.entropy_bottleneck, classifier.compressor.gaussian_conditional],
                         'Entropy bottleneck + Gaussian conditional in {}'.format(model_name))
    elif isinstance(classifier, InputCompressionClassifier):
        check_model_size(classifier.compressor.encoder, 'Encoder in {}'.format(model_name))
        check_model_size(classifier.compressor.entropy_bottleneck,
                         'Entropy bottleneck in {}'.format(model_name))
    elif isinstance(classifier, CustomResNet):
        check_model_size(classifier.bottleneck.encoder, 'Encoder in {}'.format(model_name))
    elif isinstance(classifier, BottleneckResNet):
        check_model_size(classifier.backbone.bottleneck_layer.encoder, 'Encoder in {}'.format(model_name))
        check_model_size(classifier.backbone.bottleneck_layer.entropy_bottleneck,
                         'Entropy bottleneck in {}'.format(model_name))

    # Total model size
    check_model_size(classifier, model_name)


def load_detector(model_config):
    if 'compressor' not in model_config:
        model = get_object_detection_model(model_config)
        if model is None:
            repo_or_dir = model_config.get('repo_or_dir', None)
            model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

        model_ckpt_file_path = model_config['ckpt']
        if load_bottleneck_model_ckpt(model, model_ckpt_file_path):
            return model

        load_ckpt(model_ckpt_file_path, model=model, strict=True)
        return model

    # Define compressor
    compressor_config = model_config['compressor']
    compressor = get_compression_model(compressor_config['name'], **compressor_config['params']) \
        if compressor_config is not None else None

    if compressor is not None:
        compressor_ckpt_file_path = compressor_config['ckpt']
        if os.path.isfile(compressor_ckpt_file_path):
            print('Loading compressor parameters')
            state_dict = torch.load(compressor_ckpt_file_path)
            # Old parameter keys do not work with recent version of compressai
            state_dict = load_pretrained(state_dict)
            compressor.load_state_dict(state_dict)

        print('Updating compression model')
        compressor.update()

    # Define detector
    detector_config = model_config['detector']
    detector = get_object_detection_model(detector_config)
    if detector is None:
        repo_or_dir = detector_config.get('repo_or_dir', None)
        detector = get_model(detector_config['name'], repo_or_dir, **detector_config['params'])

    detector_ckpt_file_path = detector_config['ckpt']
    load_ckpt(detector_ckpt_file_path, model=detector, strict=True)
    custom_model = get_custom_detector(model_config['name'], compressor, detector, **model_config['params'])
    return custom_model


def check_detector_size(detector, model_name):
    if isinstance(detector, InputCompressionDetector)\
            and isinstance(detector.detector.transform.compressor, MeanScaleHyperprior):
        transform = detector.detector.transform
        check_model_size([transform.compressor.g_a, transform.compressor.h_a,
                          transform.compressor.h_s], 'Encoder in {}'.format(model_name))
        check_model_size([transform.compressor.entropy_bottleneck,
                          transform.compressor.gaussian_conditional],
                         'Entropy bottleneck + Gaussian conditional in {}'.format(model_name))
    elif isinstance(detector, InputCompressionDetector):
        transform = detector.detector.transform
        check_model_size(transform.compressor.encoder, 'Encoder in {}'.format(model_name))
        check_model_size(transform.compressor.entropy_bottleneck,
                         'Entropy bottleneck in {}'.format(model_name))
    elif check_if_module_exits(detector, 'backbone.body.layer1.encoder'):
        check_model_size([detector.backbone.body.conv1, detector.backbone.body.bn1,
                          detector.backbone.body.layer1.encoder], 'Encoder in {}'.format(model_name))
    elif check_if_module_exits(detector, 'backbone.body.bottleneck_layer'):
        check_model_size(detector.backbone.body.bottleneck_layer.encoder, 'Encoder in {}'.format(model_name))
        check_model_size(detector.backbone.body.bottleneck_layer.entropy_bottleneck,
                         'Entropy bottleneck in {}'.format(model_name))

    # Total model size
    check_model_size(detector, model_name)


def load_segmenter(model_config):
    if 'compressor' not in model_config:
        model = get_semantic_segmentation_model(model_config)
        if model is None:
            repo_or_dir = model_config.get('repo_or_dir', None)
            model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

        model_ckpt_file_path = model_config['ckpt']
        if load_bottleneck_model_ckpt(model, model_ckpt_file_path):
            return model

        load_ckpt(model_ckpt_file_path, model=model, strict=True)
        return model

    # Define compressor
    compressor_config = model_config['compressor']
    compressor = get_compression_model(compressor_config['name'], **compressor_config['params'])
    compressor_ckpt_file_path = compressor_config['ckpt']
    if os.path.isfile(compressor_ckpt_file_path):
        print('Loading compressor parameters')
        state_dict = torch.load(compressor_ckpt_file_path)
        # Old parameter keys do not work with recent version of compressai
        state_dict = load_pretrained(state_dict)
        compressor.load_state_dict(state_dict)

    print('Updating compression model')
    compressor.update()
    # Define segmenter
    segmenter_config = model_config['segmenter']
    segmenter = get_semantic_segmentation_model(segmenter_config)
    if segmenter is None:
        repo_or_dir = segmenter_config.get('repo_or_dir', None)
        segmenter = get_model(segmenter_config['name'], repo_or_dir, **segmenter_config['params'])

    segmenter_ckpt_file_path = segmenter_config['ckpt']
    load_ckpt(segmenter_ckpt_file_path, model=segmenter, strict=True)
    custom_model = get_custom_segmenter(model_config['name'], compressor, segmenter, **model_config['params'])
    return custom_model


def check_segmenter_size(segmenter, model_name):
    if isinstance(segmenter, InputCompressionSegmenter)\
            and isinstance(segmenter.compressor, MeanScaleHyperprior):
        check_model_size([segmenter.compressor.g_a, segmenter.compressor.h_a,
                          segmenter.compressor.h_s], 'Encoder in {}'.format(model_name))
        check_model_size([segmenter.compressor.entropy_bottleneck,
                          segmenter.compressor.gaussian_conditional],
                         'Entropy bottleneck + Gaussian conditional in {}'.format(model_name))
    elif isinstance(segmenter, InputCompressionSegmenter):
        check_model_size(segmenter.compressor.encoder, 'Encoder in {}'.format(model_name))
        check_model_size(segmenter.compressor.entropy_bottleneck,
                         'Entropy bottleneck in {}'.format(model_name))
    elif check_if_module_exits(segmenter, 'backbone.layer1.encoder'):
        check_model_size([segmenter.backbone.conv1, segmenter.backbone.bn1,
                          segmenter.backbone.layer1.encoder], 'Encoder in {}'.format(model_name))
    elif check_if_module_exits(segmenter, 'backbone.bottleneck_layer'):
        check_model_size(segmenter.backbone.bottleneck_layer.encoder, 'Encoder in {}'.format(model_name))
        check_model_size(segmenter.backbone.bottleneck_layer.entropy_bottleneck,
                         'Entropy bottleneck in {}'.format(model_name))

    # Total model size
    check_model_size(segmenter, model_name)


def main(args):
    torchvision_model_name = args.model_name
    classifier_config_file_path = args.classifier
    detector_config_file_path = args.detector
    segmenter_config_file_path = args.segmenter
    if torchvision_model_name is not None:
        model = OFFICIAL_MODEL_DICT[torchvision_model_name](pretrained=False)
        check_model_size(model, torchvision_model_name)

    if classifier_config_file_path is not None:
        model_config = get_model_config(classifier_config_file_path)
        classifier = load_classifier(model_config)
        check_classifier_size(classifier, model_config['name'])

    if detector_config_file_path is not None:
        model_config = get_model_config(detector_config_file_path)
        detector = load_detector(model_config)
        check_detector_size(detector, model_config['name'])

    if segmenter_config_file_path is not None:
        model_config = get_model_config(segmenter_config_file_path)
        segmenter = load_segmenter(model_config)
        check_segmenter_size(segmenter, model_config['name'])


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
