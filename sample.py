import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import re
import glob
import click
import torch
from PIL import Image
import numpy as np
from typing import List


from tqdm import trange
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CLIPの読み込み
from CLIP.clip import clip
model_clip, preprocess_clip = clip.load('ViT-B/32', jit=True)  
model_clip = model_clip.eval()

# 前処理設定
preprocess_clip = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor()
])
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)



def compute_clip_loss(img, text):
    # img = clip_transform(img)
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)
    img_logits, _text_logits = model_clip(img, tokenized_text)
    return 1/img_logits * 100




# StyleGAN2-ADAの読み込み
import StyleGAN2.dnnlib as dnnlib
from StyleGAN2 import legacy



network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    generator = legacy.load_network_pkl(f)['G_ema'].to(device)



latent_shape = (1, 1, 512)
latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
latents = torch.nn.Parameter(latents_init, requires_grad=True)


lr = 1e-2 
optimizer = torch.optim.Adam(
    params=[latents],
    lr=lr,
    betas=(0.9, 0.999),
)


text = "She is a charming woman with blonde hair and blue eyes"

max_iter = 101
counter = 0

for i in trange(max_iter):
    dlatents = latents.repeat(1,18,1)
    image = generator.synthesis(dlatents)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


    loss = compute_clip_loss(image, text) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./{i:04d}.png')