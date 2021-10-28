import torch
from CLIP.clip import clip

class CLIPLoss(torch.nn.Module):

    def __init__(self, model):
        super(CLIPLoss, self).__init__()
        self.model = model
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_l2_loss = torch.nn.MSELoss().to(self.device).eval()


    def forward(self, image, text, w, w_hat):
        image = self.avg_pool(self.upsample(image))
        tokenized_text = clip.tokenize([text]).to(self.device)
        similarity = 1 - self.model(image, tokenized_text)[0] / 100

        loss_l2_latent = self.latent_l2_loss(w_hat, w)

        loss = similarity + loss_l2_latent
        return loss