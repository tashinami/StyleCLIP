import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CLIPの読み込み
import clip
model, preprocess = clip.load('ViT-B/32', jit=True)  
model = model.eval()  


# 前処理設定
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor()
])
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)


# 画像の読み込み
images =[]
files = glob.glob('./img/*.png')
files.sort()

print("画像の前処理...")
for i, file in enumerate(tqdm(files)):
      image = preprocess(Image.open(file).convert("RGB"))
      images.append(image)

image_input = torch.tensor(np.stack(images)).to(device)
image_input -= image_mean[:, None, None]
image_input /= image_std[:, None, None]


# 検索文字列の前処理
text = 'She is a charming woman with blonde hair and blue eyes'
text_input = clip.tokenize(text)
text_input = text_input.to(device)


# --- 画像とテキストのCOS類似度の計算 ----
print("特徴ベクトルを抽出...")
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_input).float()
    text_features /= text_features.norm(dim=-1, keepdim=True) 

# COS類似度を計算
print("類似度算出...")
text_probs = torch.cosine_similarity(image_features, text_features)

# COS類似度の高い順にインデックスをソート
x = np.argsort(-text_probs.cpu(), axis=0)

# COS類似度TOP10を表示
fig = plt.figure(figsize=(30, 40))
for i in range(10):
    name = str(x[i].item()).zfill(6)+'.png'
    img = Image.open('./img/'+name)    
    images = np.asarray(img)
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
    image_plt = np.array(images)
    ax.imshow(image_plt)
    cos_value = round(text_probs[x[i].item()].item(), 3)
    ax.set_xlabel(cos_value, fontsize=12)               
plt.show()
plt.close()  