from torch import nn
from torchdistill.losses.single import register_single_loss


@register_single_loss
class BppBasedLoss(nn.Module):
    def __init__(self, compressor_module_path, beta, reduction='mean'):
        super().__init__()
        self.compressor_module_path = compressor_module_path
        self.beta = beta
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, student_io_dict, *args, **kwargs):
        compressor_module_dict = student_io_dict[self.compressor_module_path]
        compressor_inputs = compressor_module_dict['input']
        reconstructed_inputs, likelihoods = compressor_module_dict['output']
        reconstruction_loss = self.mse(reconstructed_inputs, compressor_inputs)
        n, _, h, w = compressor_inputs.shape
        num_pixels = n * h * w
        bpp = -likelihoods.log2().sum() if self.reduction == 'sum' else -likelihoods.log2().sum() / num_pixels
        return reconstruction_loss + self.beta * bpp


@register_single_loss
class BppLoss(nn.Module):
    def __init__(self, entropy_module_path, reduction='mean'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction

    def forward(self, student_io_dict, *args, **kwargs):
        entropy_module_dict = student_io_dict[self.entropy_module_path]
        reconstructed_inputs, likelihoods = entropy_module_dict['output']
        bpp = -likelihoods.log2().sum()
        return bpp
