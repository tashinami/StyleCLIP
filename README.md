# StyleCLIP
テキストで人の顔画像をコントロールするStyleCLIPの実装です。
https://arxiv.org/abs/2103.17249

## Requirement
- pytorch
- ftfy
- ninja

## Usage
### 顔画像からStyleGAN2の潜在空間を取得する
```
python image2latent.py

```


### テキストに合わせた顔画像を生成する
```
python text2face.py --itr 301 --text 'She is a charming woman with blonde hair and blue eyes'  --seed 85

```
