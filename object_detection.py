import argparse
import builtins as __builtin__
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
from compressai.zoo.pretrained import load_pretrained
from torch import distributed as dist
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate
from torchdistill.common import file_util, module_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.datasets.coco import get_coco_api_from_dataset
from torchdistill.eval.coco import CocoEvaluator
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.official import get_object_detection_model
from torchdistill.models.registry import get_model
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN

from compression.registry import get_compression_model
from custom.detector import InputCompressionDetector, get_custom_model
from custom.util import check_if_module_exits, load_bottleneck_model_ckpt, extract_entropy_bottleneck_module

logger = def_logger.getChild(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for object detection models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('-test_only', action='store_true', help='Only test the models')
    parser.add_argument('-student_only', action='store_true', help='Test the student model only')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    parser.add_argument('-warm_up', action='store_true',
                        help='use warm up strategy for the first epoch')
    return parser


def load_model(model_config, device):
    if 'compressor' not in model_config:
        model = get_object_detection_model(model_config)
        if model is None:
            repo_or_dir = model_config.get('repo_or_dir', None)
            model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

        model_ckpt_file_path = os.path.expanduser(model_config['ckpt'])
        if load_bottleneck_model_ckpt(model, model_ckpt_file_path):
            return model.to(device)

        load_ckpt(model_ckpt_file_path, model=model, strict=True)
        return model.to(device)

    # Define compressor
    compressor_config = model_config['compressor']
    compressor = get_compression_model(compressor_config['name'], **compressor_config['params']) \
        if compressor_config is not None else None

    if compressor is not None:
        compressor_ckpt_file_path = os.path.expanduser(compressor_config['ckpt'])
        if os.path.isfile(compressor_ckpt_file_path):
            logger.info('Loading compressor parameters')
            state_dict = torch.load(compressor_ckpt_file_path)
            # Old parameter keys do not work with recent version of compressai
            state_dict = load_pretrained(state_dict)
            compressor.load_state_dict(state_dict)

        logger.info('Updating compression model')
        compressor.update()

    # Define detector
    detector_config = model_config['detector']
    detector = get_object_detection_model(detector_config)
    if detector is None:
        repo_or_dir = detector_config.get('repo_or_dir', None)
        detector = get_model(detector_config['name'], repo_or_dir, **detector_config['params'])

    detector_ckpt_file_path = os.path.expanduser(detector_config['ckpt'])
    load_ckpt(detector_ckpt_file_path, model=detector, strict=True)
    custom_model = get_custom_model(model_config['name'], compressor, detector, **model_config['params'])
    return custom_model.to(device)


def train_one_epoch(training_box, bottleneck_updated, warms_up, device, epoch, log_freq):
    model = training_box.student_model if hasattr(training_box, 'student_model') else training_box.model
    entropy_bottleneck_module = extract_entropy_bottleneck_module(model)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    warm_up_lr_scheduler = None
    if warms_up and epoch == 0:
        logger.info('Setting up warm up lr scheduler')
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(training_box.train_data_loader) - 1)

        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        warm_up_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(training_box.optimizer, f)

    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        supp_dict = default_collate(supp_dict)
        loss = training_box(sample_batch, targets, supp_dict)
        aux_loss = None
        if isinstance(entropy_bottleneck_module, nn.Module) and not bottleneck_updated:
            aux_loss = entropy_bottleneck_module.aux_loss()
            aux_loss.backward()

        training_box.update_params(loss)
        if warm_up_lr_scheduler is not None:
            warm_up_lr_scheduler.step()

        batch_size = len(sample_batch)
        loss_value = loss.item()
        if aux_loss is None:
            if not math.isfinite(loss_value):
                logger.info('Loss is {}, stopping training'.format(loss_value))
                sys.exit(1)
            metric_logger.update(loss=loss_value, lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            aux_loss_value = aux_loss.item()
            if not math.isfinite(aux_loss_value):
                logger.info('Aux loss is {}, stopping training'.format(aux_loss_value))
                sys.exit(1)
            metric_logger.update(loss=loss_value, aux_loss=aux_loss_value,
                                 lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, DistributedDataParallel):
        model_without_ddp = model.module

    if isinstance(model_without_ddp, InputCompressionDetector):
        model_without_ddp = model_without_ddp.detector

    iou_type_list = ['bbox']
    if isinstance(model_without_ddp, MaskRCNN):
        iou_type_list.append('segm')
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_type_list.append('keypoints')
    return iou_type_list


def log_info(*args, **kwargs):
    force = kwargs.pop('force', False)
    if is_main_process() or force:
        logger.info(*args, **kwargs)


@torch.no_grad()
def evaluate(model, data_loader, device, device_ids, distributed, bottleneck_updated=False,
             log_freq=1000, title=None, header='Test:'):
    model.to(device)
    entropy_bottleneck_module = extract_entropy_bottleneck_module(model)
    if entropy_bottleneck_module is not None:
        entropy_bottleneck_module.file_size_list.clear()
        if not bottleneck_updated:
            logger.info('Updating entropy bottleneck')
            entropy_bottleneck_module.update()
    else:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    # Replace built-in print function with logger.info to log summary printed by pycocotools
    builtin_print = __builtin__.print
    __builtin__.print = log_info

    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for sample_batch, targets in metric_logger.log_every(data_loader, log_freq, header):
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(sample_batch)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_stats_str = 'Averaged stats: {}'.format(metric_logger)
    logger.info(avg_stats_str)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Revert print function
    __builtin__.print = builtin_print

    torch.set_num_threads(n_threads)
    return coco_evaluator


def train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args):
    logger.info('Start distillation')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_map = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_map, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    entropy_bottleneck_module = extract_entropy_bottleneck_module(student_model_without_ddp)
    epoch_to_update = train_config.get('epoch_to_update', None)
    bottleneck_updated = False
    warms_up = args.warm_up
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        if epoch_to_update is not None and epoch_to_update <= epoch and not bottleneck_updated:
            logger.info('Updating entropy bottleneck')
            student_model_without_ddp.backbone.body.bottleneck_layer.update()
            bottleneck_updated = True

        train_one_epoch(training_box, bottleneck_updated, warms_up, device, epoch, log_freq)
        if entropy_bottleneck_module is None or bottleneck_updated:
            val_coco_evaluator =\
                evaluate(student_model, training_box.val_data_loader, device, device_ids, distributed,
                         bottleneck_updated, log_freq=log_freq, header='Validation:')
            # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
            val_map = val_coco_evaluator.coco_eval['bbox'].stats[0]
            if val_map > best_val_map and is_main_process():
                logger.info('Updating ckpt (Best BBox mAP: {:.4f} -> {:.4f})'.format(best_val_map, val_map))
                best_val_map = val_map
                save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                          best_val_map, config, args, ckpt_file_path)
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    if entropy_bottleneck_module is not None:
        save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                  best_val_map, config, args, ckpt_file_path)
    training_box.clean_modules()


def analyze_bottleneck_size(model):
    file_size_list = list()
    if isinstance(model, InputCompressionDetector):
        file_size_list = model.detector.transform.file_size_list
    elif check_if_module_exits(model, 'backbone.body.layer1.compressor')\
            and model.backbone.body.layer1.compressor is not None:
        file_size_list = model.backbone.body.layer1.compressor.file_size_list
    elif check_if_module_exits(model, 'backbone.body.bottleneck_layer'):
        file_size_list = model.backbone.body.bottleneck_layer.file_size_list

    if len(file_size_list) == 0:
        return

    file_sizes = np.array(file_size_list)
    logger.info('Bottleneck size [KB]: mean {} std {} for {} samples'.format(file_sizes.mean(), file_sizes.std(),
                                                                             len(file_sizes)))


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    cudnn.benchmark = True
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    device = torch.device(args.device)
    dataset_dict = util.get_all_dataset(config['datasets'])
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model = load_model(teacher_model_config, device) if teacher_model_config is not None else None
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    student_model = load_model(student_model_config, device)
    if not args.test_only:
        ckpt_file_path = student_model_config['ckpt']
        train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args)
        student_model_without_ddp =\
            student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, device, device_ids, distributed, bottleneck_updated=False,
                 title='[Teacher: {}]'.format(teacher_model_config['name']))
    evaluate(student_model, test_data_loader, device, device_ids, distributed, bottleneck_updated=False,
             title='[Student: {}]'.format(student_model_config['name']))
    analyze_bottleneck_size(student_model)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
