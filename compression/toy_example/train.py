from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from data import load_data
from config import device
import torch
import os
import config
import torch.nn.functional as F
from toy_model import ImgModel

tbdir = config.tbdir
imgbeta = config.beta
grad_clip = config.grad_clip
img_data, img_test = load_data(config.data_config, config.batch_size, pin_memory=False, num_workers=8, sequence=False)
dataset_name = config.data_config['dataset_name']
model_name = f'toyimg-{imgbeta}'
step = 0

writer = SummaryWriter(f'{tbdir}/{model_name}')

imgmodel = ImgModel(config.filter_size)

if os.path.exists(f'./model_params/{model_name}.pt'):
    loaded = torch.load(f'./model_params/{model_name}-epoch.pt', map_location=lambda storage, loc: storage)
    imgmodel.load_state_dict(loaded)

imgmodel = imgmodel.to(device)
optimizer = optim.Adam([{
    'params': imgmodel.parameters(),
    'lr': config.lr
}, {
    'params': imgmodel.aux_parameters(),
    'lr': config.lr * 10
}])

if os.path.exists(f'./model_params/{model_name}-opt.pt'):
    loaded = torch.load(f'./model_params/{model_name}-opt.pt', map_location=lambda storage, loc: storage)
    optimizer.load_state_dict(loaded)
    step = torch.load(f'./model_params/{model_name}-step.pt')


def schedule_func(ep):
    return max(config.decay**ep, config.minf)


scheduler = LambdaLR(optimizer, lr_lambda=[schedule_func for i in range(2)])


def get_batch_psnr(m1, m2, max_val=255):
    assert (m1.shape == m2.shape)
    N, C, H, W = m1.shape
    mse = ((m1 - m2)**2).view(N, -1).mean(1)
    return (10 * torch.log10((max_val**2) / mse)).mean()


def compute_loss(recon, likelihood, target, beta):
    N, _, H, W = target.shape
    num_pixels = N * H * W
    bpp = -likelihood.log2().sum() / num_pixels
    mse = F.mse_loss(recon, target)
    loss = mse + beta * bpp
    psnr = get_batch_psnr(recon.clamp(0, 1), target, 1)
    return loss, psnr, bpp


def train_image(batch, beta):
    imgmodel.train()
    optimizer.zero_grad()
    img, likelihood = imgmodel(batch)
    loss, psnr, bpp = compute_loss(img, likelihood, batch, beta)
    extra_loss = imgmodel.aux_loss()
    loss.backward()
    extra_loss.backward()
    torch.nn.utils.clip_grad_norm_(imgmodel.parameters(), config.grad_clip)
    optimizer.step()
    return img, bpp, psnr


def test_image(batch, beta):
    with torch.no_grad():
        imgmodel.eval()
        img, likelihood = imgmodel(batch)
        loss, psnr, bpp = compute_loss(img, likelihood, batch, beta)
        return img, bpp, psnr


checkpoint = 2000
finish = False
while True:
    for train_batch_idx, train_batch in enumerate(img_data):
        imgs, bpp, psnr = train_image(train_batch.to(device), imgbeta)
        if step % checkpoint == 0:
            writer.add_images('train/reconstruct', imgs, step // checkpoint)
            writer.add_scalar('train/bpp_per_batch_per_frame', bpp.mean().item(), step // checkpoint)
            writer.add_scalar('train/psnr_per_batch_per_frame', psnr.mean().item(), step // checkpoint)
            test_bpp, test_psnr = 0, 0
            for test_batch_idx, test_batch in enumerate(img_test):
                imgs, bpp, psnr = test_image(test_batch.to(device), imgbeta)
                if test_batch_idx == 0:
                    writer.add_images('test/reconstruct', imgs, step // checkpoint)
                test_bpp += bpp.item() * test_batch.shape[0]
                test_psnr += psnr.item() * test_batch.shape[0]
            writer.add_scalar('test/bpp_per_val_datum_per_frame', test_bpp / len(img_test.dataset), step // checkpoint)
            writer.add_scalar('test/psnr_per_val_datum_per_frame', test_psnr / len(img_test.dataset),
                              step // checkpoint)
        step += 1
        if step % config.save_step == 0:
            torch.save(imgmodel.state_dict(), f'./model_params/{model_name}-epoch.pt')
            torch.save(optimizer.state_dict(), f'./model_params/{model_name}-opt.pt')

        if step > (config.n_step + config.extra_step):
            finish = True
            break

        if (step % config.scheduler_step == 0) and (step != 0):
            scheduler.step()
    if finish:
        break

torch.save(imgmodel.state_dict(), f'./model_params/{model_name}-epoch.pt')
torch.save(optimizer.state_dict(), f'./model_params/{model_name}-opt.pt')
