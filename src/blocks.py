import torch.nn as nn
import torch

class ResBlock(nn.Module):
    """
    # Residual Block

    This Residual Block contains

    ## Architecture
    - `Conv2d`: A convolutional layer with the given kernel size and stride. Takes in `inChannels` channels and outputs `outChannels` channels
    - `BatchNorm2d`: Batch normalization for the output channels
    - `PReLU`: Parametric ReLU activation function
    - `Conv2d`: A convolutional layer with the given kernel size and stride. Takes in `outChannels` channels and outputs `outChannels` channels
    - `BatchNorm2d`: Batch normalization for the output channels

    After passing through the block, the residual block adds the input to the output of the blocks and returns it.
    """
    def __init__(self, inChannels:int, outChannels:int, kernelSize:int, stride:int, *args, **kwargs) -> None:
        """
        Initializes a Res Block

        Parameters:
        inChannels (int): Number of input channels
        outChannels (int): Number of output channels
        kernelSize (int): Kernel size of the convolutional layers
        stride (int): Stride of the convolutional layers
        """
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=stride, padding="same"),
            nn.BatchNorm2d(outChannels),
            nn.PReLU(),
            nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, stride=stride, padding="same"),
            nn.BatchNorm2d(outChannels)
        )

    def forward(self, x):
        return self.block(x) + x
    
class Upscale2xBlock(nn.Module):
    """
    # Upscale 2x Block

    Upscales the given input by a factor of 2

    ## Architecture
    - `Conv2d`: A convolutional layer with `3` kernel size and `1` stride. Takes in `inChannels` channels and outputs `outChannels` channels
    - `PixelShuffle`: Pixel shuffle layer with `upscale_factor=2`
    - `PReLU`: Parametric ReLU activation function

    After passing through the block, the upscale block upscales the input by a factor of 2 by shuffling the channels.
    """
    def __init__(self,inChannels:int, outChannels:int, *args, **kwargs) -> None:
        """
        Initializes a Upscale 2x Block

        Parameters:
        inChannels (int): Number of input channels
        outChannels (int): Number of output channels
        """
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding="same"),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.block(x)

class DiscriminatorBlock(nn.Module):
    """
    # Discriminator Block

    A block used inside the discriminator model.

    ## Architecture
    - `Conv2d`: A convolutional layer with the given kernel size and stride. Takes in `inChannels` channels and outputs `outChannels` channels
    - `BatchNorm2d`: Batch normalization for the output channels
    - `LeakyReLU`: Leaky ReLU activation function with `negative_slope=0.2`
    """
    def __init__(self, inChannels:int, outChannels:int, kernelSize:int, stride:int, useBatchNorm:bool=True, *args, **kwargs) -> None:
        """
        Initializes a Disscriminator BLock

        Parameters:
        inChannels (int): Number of input channels
        outChannels (int): Number of output channels
        kernelSize (int): Kernel size of the convolutional layers
        stride (int): Stride of the convolutional layers
        useBatchNorm (bool): Whether to use batch normalization or not
        """
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=stride, padding=1),
            nn.BatchNorm2d(outChannels)if useBatchNorm else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class MiDaSBlock(nn.Module):
    """
    # MiDaS Block

    MiDas Block calculates the depth of the given image.
    """
    def __init__(self, midas:nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.midas = midas

    def forward(self, x):
        with torch.no_grad():
            features = self.midas(x)
            # getting th depth map 

            minValue = features.min()
            maxValue = features.max()
            depth = (features - minValue) / (maxValue - minValue)
            depth = (depth - 0.5) * 2
            # normalizing depth 
                        
            depth = depth.unsqueeze(1)

            return depth