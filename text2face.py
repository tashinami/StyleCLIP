import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.simplefilter('ignore')

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List

from CLIP.clip import clip
from CLIPLoss import CLIPLoss

import StyleGAN2.dnnlib as dnnlib
from StyleGAN2 import legacy

from mapper.StyleCLIPMapper import StyleCLIPMapper

def tensor2cvImage(image):
    '''
      OpenCV形式の画像に変換する
    '''
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = image[0].cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    return image


def arg_parse():
    '''
      各種パラメータの読み込み
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--itr', default=151, type=int)
    parser.add_argument('--pretrained_model', default="ffhq", type=str)

    parser.add_argument('--image_path', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--latent_path', default=None, type=str)
    parser.add_argument('--text', default="blue eyes", type=str)

    parser.add_argument('--out_dir', default="./result", type=str)

    parser.add_argument('--is_save_movie', default=False, type=bool)
    parser.add_argument('--save_freq', default=10, type=int)


    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
    args = parser.parse_args()
    return args


def styleGAN2_model_zoo(args):
    '''
      StyleGAN2の事前学習モデルのURLを取得する
    '''
    dataset = args.pretrained_model

    if dataset == "ffhq":
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

    elif dataset == "metface":
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl"

    elif dataset == "cifar10":
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl"

    else:
        url = args.pretrained_model

    return url



if __name__ == "__main__":
    args = arg_parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CLIPの読み込み
    model_clip, preprocess_clip = clip.load('ViT-B/32', jit=True)  
    model_clip = model_clip.eval()

    # ロス
    clip_loss = CLIPLoss(model_clip)

    # StyleGAN2-ADAの読み込み
    network_pkl = styleGAN2_model_zoo(args)
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device)

    # 潜在変数をセット
    # 乱数シードによる指定
    if args.seed is not None:
      z = np.random.RandomState(args.seed).randn(1, generator.z_dim)
      latents_init = generator.mapping(torch.from_numpy(z).to(device), None)
      latents = latents_init[:, :1, :].cpu().numpy().astype(np.float32)
      latents = torch.from_numpy(latents).to(device)
      latents = torch.nn.Parameter(latents, requires_grad=True)

    # 潜在変数を直接指定
    elif args.latent_path is not None:
      latents_init = np.load(args.latent_path)['w']
      latents = latents_init[:, :1, :].astype(np.float32)
      latents = torch.from_numpy(latents).to(device)
      latents = torch.nn.Parameter(latents, requires_grad=True)

    # 全てゼロの潜在変数を指定
    else:
      latent_shape = (1, 1, 512)
      latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
      latents = torch.nn.Parameter(latents_init, requires_grad=True)


    # 潜在変数のマッパー
    mapper = StyleCLIPMapper()

    # 学習パラメータ
    lr = 1e-2 
    max_iter = args.itr
    optimizer = torch.optim.Adam(params=mapper.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, verbose=True)


    # 出力ディレクトリを作成
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(os.path.join(out_dir, "out.mp4"), fourcc, 30, (512, 512))

    # 学習スタート
    noise_mode = 'const'
    with tqdm(total=max_iter, unit="itr") as pbar:
        for i in range(max_iter):
            w = latents
            w_hat = w + 0.1 * mapper(w)
            dlatents = w_hat.repeat(1,18,1)
            image = generator.synthesis(dlatents, noise_mode=noise_mode)
            loss = clip_loss(image, args.text, w, w_hat) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            pbar.set_postfix({"loss":loss.item()})
            pbar.update(1)

            image = tensor2cvImage(image)
            writer.write(image)

            if i % 10 == 0:
                cv2.imwrite(os.path.join(out_dir, str(i).zfill(4) + ".png"), image)


