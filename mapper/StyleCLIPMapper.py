import torch
from torch import nn
from mapper.latent_mapper import SingleMapper, LevelsMapper

class StyleCLIPMapper(nn.Module):
    def __init__(self):
        super(StyleCLIPMapper, self).__init__()

        self.mapper = LevelsMapper()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mapper = self.mapper.to(device)

    def load_weights(self, checkpoint_path):
        if checkpoint_path is not None:
            print('Loading from  mapper checkpoint: {}'.format(checkpoint_path))
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.mapper.load_state_dict(ckpt, strict=True)

    def forward(self, x, input_code=False):
        if input_code:
            codes = x
        else:
            codes = self.mapper(x)

        return codes
