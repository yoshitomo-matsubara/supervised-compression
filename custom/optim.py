import torch
from torchdistill.common.func_util import register_scheduler


@register_scheduler
def custom_lambda_lr(optimizer, num_iterations, num_epochs, factor=0.9):
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (num_iterations * num_epochs)) ** factor)
    return lr_scheduler
