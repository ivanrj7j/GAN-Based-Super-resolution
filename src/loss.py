import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights
import config

class VGGLoss(nn.Module):
    """
    VGG loss using pretrained vgg19 model
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, yhat, y):
        return self.loss.forward(self.vgg(yhat), self.vgg(y))