import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.simplefilter('ignore')

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import StyleGAN2.dnnlib as dnnlib
from StyleGAN2 import legacy

def tensor2cvImage(image):
    '''
      OpenCV形式の画像に変換する
    '''
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image[0].cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    return image

seed = 300
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# StyleGAN2-ADAの読み込み
network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    generator = legacy.load_network_pkl(f)['G_ema'].to(device)

# Load target image.
target_image = cv2.imread("image.png")
target_image = cv2.resize(target_image, (generator.img_resolution, generator.img_resolution))
target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
target = torch.tensor(target_image.transpose([2, 0, 1]), device=device)


num_steps = 151
w_avg_samples = 10000
initial_learning_rate = 0.1

initial_noise_factor = 0.5
lr_rampdown_length         = 0.25
lr_rampup_length           = 0.05
noise_ramp_length          = 0.75
regularize_noise_weight    = 1e5
verbose                    = False

# Compute w stats.
z_samples = np.random.RandomState(123).randn(w_avg_samples, generator.z_dim)
w_samples = generator.mapping(torch.from_numpy(z_samples).to(device), None)
w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
w_avg = np.mean(w_samples, axis=0, keepdims=True)
w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5


# Setup noise inputs.
noise_bufs = { name: buf for (name, buf) in generator.synthesis.named_buffers() if 'noise_const' in name }

# Load VGG16 feature detector.
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)

# Features for target image.
target_images = target.unsqueeze(0).to(device).to(torch.float32)
if target_images.shape[2] > 256:
    target_images = F.interpolate(target_images, size=(256, 256), mode='area')
target_features = vgg16(target_images, resize_images=False, return_lpips=True)

w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

# Init noise.
for buf in noise_bufs.values():
    buf[:] = torch.randn_like(buf)
    buf.requires_grad = True

# search latent from image
with tqdm(total=num_steps, unit="itr") as pbar:
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, generator.mapping.num_ws, 1])
        synth_images = generator.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"dist":dist.item(), "loss" : loss.item()})
        pbar.update(1)

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        results =  w_out.repeat([1, generator.mapping.num_ws, 1])


# 出力ディレクトリを作成
out_dir = "latent"
os.makedirs(out_dir, exist_ok=True)

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(os.path.join(out_dir, "out.mp4"), fourcc, 30, (512, 512))

# 動画作成
print("Video output...")
for projected_w in tqdm(results):
    image = generator.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    image = tensor2cvImage(image)
    writer.write(image)

# 潜在変数を保存
print("Latent output...")
projected_w = results[-1]
os.path.join(out_dir, "projected_w.npz")
np.savez(os.path.join(out_dir, "projected_w.npz"), w=projected_w.unsqueeze(0).cpu().numpy())

print("Finish !!")









