import torch

class CLIPLoss(torch.nn.Module):

    def __init__(self, model):
        super(CLIPLoss, self).__init__()
        self.model = model
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity