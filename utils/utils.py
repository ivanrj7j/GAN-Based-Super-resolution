import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from src import Generator, Discriminator
import config

def calculatePSNR(originalImage:torch.Tensor, generatedImage:torch.Tensor):
    """
    Calculates the Peak Signal to Noise Ratio between the original and generated image.
    Input should be in the range [0, 1]

    Parameters:
    originalImage (torch.Tensor): Original image tensor
    generatedImage (torch.Tensor): Generated image tensor
    """
    with torch.no_grad():
        mse = torch.mean(torch.square(originalImage-generatedImage))
        return 10 * torch.log10(1 / mse)

def generateImage(originalImage:torch.Tensor, generator:Generator):
    """
    Generates the image using generator

    Parameters:
    originalImage (torch.Tensor): Original image tensor
    generator (Generator): Generator model
    """
    with torch.no_grad():
        output = (generator.forward(originalImage)+1)/2

        return output

def writeSummary(writer:SummaryWriter, inputBatch:torch.Tensor, upscaledImages:torch.Tensor, generator:Generator, losses:dict[str, float], epoch:int):
    """
    Writes the input batch, generated images, and loss to TensorBoard.

    Parameters:
    writer (SummaryWriter): TensorBoard SummaryWriter object
    inputBatch (torch.Tensor): Input batch tensor
    upscaledImages (torch.Tensor): Original image with higher resolution `Not Genrated Images`.
    generator (Generator): Generator model
    losses (dict[str, float]): Dictionary containing loss values for different components
    epoch (int): Current epoch number
    """
    images = generateImage(inputBatch, generator)
    upscaledImages = (upscaledImages + 1)/2
    psnr = calculatePSNR(upscaledImages, images)
    
    writer.add_image('Original', make_grid(upscaledImages), global_step=epoch, )
    writer.add_image('Generated Image', make_grid(images), global_step=epoch, )
    writer.add_scalar('PSNR', psnr, global_step=epoch)
    for loss in losses:
        writer.add_scalar(loss, losses[loss], global_step=epoch)

def loadModels(generatorPath:str, discriminatorPath:str, midas:torch.nn.Module=None, imageChannels:int=3, scale=4, device:str="cuda"):
    """
    Loads the generator and discriminator from the given paths if paths are provided, else loads a simple model

    Parameters:
    generatorPath (str): Path to the saved generator model. Does not load any weights if path = ""
    discriminatorPath (str): Path to the saved discriminator model. Does not load any weights if path = ""
    midas (torch.nn.Module, optional): Midas model for feature extraction. Defaults to None.
    imageChannels (int): Number of channels in the output image. Defaults to 3.
    scale (int): Upscaling factor for the generator. Defaults to 4.
    device (str): Device to run the model on. Defaults to "cuda".
    """

    generator = Generator(midas, imageChannels, scale)
    discriminator = Discriminator(imageChannels, config.useDiscClassifier)

    if generatorPath != "":
        print(f"Loading {generatorPath}")
        genratorWeights = torch.load(generatorPath, weights_only=False)
        generator.load_state_dict(genratorWeights)
    
    if discriminatorPath != "":
        print(f"Loading {discriminatorPath}")
        discriminatorWeights = torch.load(discriminatorPath, weights_only=False)
        discriminator.load_state_dict(discriminatorWeights)


    return generator.to(device), discriminator.to(device)

def saveModels(generator:Generator, discriminator:Discriminator, savePath:str, version:str):
    """
    Saves the generator and discriminator models to the given save path with the given version

    Parameters:
    generator (Generator): Generator model to save
    discriminator (Discriminator): Discriminator model to save
    savePath (str): Path to save the models
    version (str): Version of the models

    Example:
    >>> saveModels(generator, discriminator, "path/to/save/models", "v1")

    Saves at:
    -   path/to/save/models/gen-v1.pth
    -   path/to/save/models/dis-v1.pth
    """
    if generator is not None:
        torch.save(generator.state_dict(), f"{savePath}/gen-{version}.pth")
    if discriminator is not None:
        torch.save(discriminator.state_dict(), f"{savePath}/dis-{version}.pth")

def loadMidas():
    """
    Loads midas model
    """
    return torch.hub.load(config.midasName, config.midasType).to(config.device)