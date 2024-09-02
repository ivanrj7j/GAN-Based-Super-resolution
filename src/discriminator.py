import torch.nn as nn
from src.blocks import DiscriminatorBlock

class Discriminator(nn.Module):
    def __init__(self, imageChannels:int=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        block64 = nn.Sequential(
            DiscriminatorBlock(imageChannels, 64, kernelSize=3, stride=1, useBatchNorm=False),
            DiscriminatorBlock(64, 64, kernelSize=3, stride=2)
        )
        block128 = nn.Sequential(
            DiscriminatorBlock(64, 128, kernelSize=3, stride=1),
            DiscriminatorBlock(128, 128, kernelSize=3, stride=2)
        )
        block256 = nn.Sequential(
            DiscriminatorBlock(128, 256, kernelSize=3, stride=1),
            DiscriminatorBlock(256, 256, kernelSize=3, stride=2)
        )
        block512 = nn.Sequential(
            DiscriminatorBlock(256, 512, kernelSize=3, stride=1),
            DiscriminatorBlock(512, 512, kernelSize=3, stride=2)
        )

        self.convBlocks = nn.Sequential(block64, block128, block256, block512)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        convResults = self.convBlocks.forward(x)

        return self.classifier.forward(convResults)        