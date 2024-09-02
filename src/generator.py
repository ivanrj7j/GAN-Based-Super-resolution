import torch.nn as nn
from src.blocks import Upscale2xBlock, ResBlock, MiDaSBlock
import torch
from math import log2

class Generator(nn.Module):
    def __init__(self, midas:nn.Module=None, imageChannels:int=3, scale:int=4, *args, **kwargs) -> None:
        """
        Initializes the generator

        Parmeters:
        midas (nn.Module): The MiDaS model
        imageChannels (int): Number of channels in the input image (default 3)
        scale (int): Scale factor for the output image, it should be a power of 2 (default 4).

        Example:
        >>> import torch
        >>> midasType = "MiDaS_small"
        >>> midas = torch.hub.load("intel-isl/MiDaS", midasType).to("cuda")
        >>> generator = Generator(midas)
        """
        super().__init__(*args, **kwargs)
        self.midas = midas
        inChannels = imageChannels


        if midas is not None:
            self.midas_block = MiDaSBlock(midas)
            inChannels += 1

        self.initBlock = nn.Sequential(
            nn.Conv2d(inChannels, 64, kernel_size=9, stride=1, padding="same"),
            nn.PReLU()
        )

        residualBlocks = [ResBlock(64, 64, kernelSize=3, stride=1) for _ in range(5)]
        finalResidual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64)
        )
        residualBlocks.append(finalResidual)
        self.residualBlocks = nn.Sequential(*residualBlocks)

        factor = int(log2(scale))
        self.upscaleBlocks = nn.Sequential(*(Upscale2xBlock(64, 256) for _ in range(factor)))

        self.finalLayer = nn.Conv2d(64, imageChannels, kernel_size=9, stride=1, padding="same")

    def forward(self, x):
        if self.midas is not None:
            depthMap = self.midas_block(x)
            x = torch.concat((x, depthMap), 1)
            # getting depth info 
            
        initOutput = self.initBlock.forward(x)
        # passing through the first block 
        
        residualOutput = self.residualBlocks.forward(initOutput)
        residualOutput += initOutput
        # passing through the residual blocks 

        upscaleOutput = self.upscaleBlocks.forward(residualOutput)
        # upscaling the image 
        
        finalOutput = self.finalLayer.forward(upscaleOutput)

        return torch.tanh(finalOutput)
        # applying tanh to the final output is not in the paper 