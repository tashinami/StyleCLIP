import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.simplefilter('ignore')

import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from typing import List

# openCV形式の画像に変換する
def tensor2cvImage(image):
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image[0].cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    return image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CLIPの読み込み
from CLIP.clip import clip
model_clip, preprocess_clip = clip.load('ViT-B/32', jit=True)  
model_clip = model_clip.eval()


def compute_clip_loss(img, text):
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)
    img_logits, _text_logits = model_clip(img, tokenized_text)
    return 1.0 / img_logits * 100.0


# StyleGAN2-ADAの読み込み
import StyleGAN2.dnnlib as dnnlib
from StyleGAN2 import legacy

network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    generator = legacy.load_network_pkl(f)['G_ema'].to(device)



# 
latent_shape = (1, 1, 512)
latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
latents = torch.nn.Parameter(latents_init, requires_grad=True)


# 学習パラメータ
lr = 1e-2 
max_iter = 151
optimizer = torch.optim.SGD(
    params=[latents],
    lr=lr,
    momentum=0.9, nesterov=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


# 入力テキスト
text = "She is a charming woman with blonde hair and blue eyes"

# 出力ディレクトリを作成
out_dir = "result"
os.makedirs(out_dir, exist_ok=True)

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter("./" + out_dir + "/out.mp4", fourcc, 30, (512, 512))


# 学習スタート
noise_mode = 'const'
with tqdm(total=max_iter, unit="itr") as pbar:
    for i in range(max_iter):
        dlatents = latents.repeat(1,18,1)
        image = generator.synthesis(dlatents, noise_mode=noise_mode)
        loss = compute_clip_loss(image, text) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        pbar.set_postfix({"loss":loss.item()})
        pbar.update(1)

        image = tensor2cvImage(image)
        writer.write(image)

        if i % 10 == 0:
            cv2.imwrite("./" + out_dir + "/" + str(i).zfill(4) + ".png", image)