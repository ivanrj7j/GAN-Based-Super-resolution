from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import albumentations as A

class SuperResolutionDataset(Dataset):
    """
    # Super Resolution Dataset

    Super Resolution Dataste takes in a directory with high resolution images
    """
    def __init__(self, path:str, scale:int=4, resolution:int=512, assumeResolution:bool=True) -> None:
        super().__init__()
        self.path = path
        self.images = os.listdir(self.path)
        self.scale = scale
        self.resolution = resolution
        self.assumeResolution = assumeResolution

        ogTransforms = [transforms.Resize((self.resolution, self.resolution))]
        if self.assumeResolution:
            ogTransforms = []
        ogTransforms += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.ogTransforms = transforms.Compose(ogTransforms)

        self.albumentationTransforms = A.Compose([
            A.Resize(self.resolution//self.scale, self.resolution//self.scale),
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.ImageCompression(90, 100, p=1),
            A.Blur(p=1),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def downscaleImage(self, image):
        image = np.array(image)
        image = self.albumentationTransforms(image=image)["image"]

        return transforms.ToTensor()(image)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        imagePath = os.path.join(self.path, self.images[index])
        image = Image.open(imagePath)
        
        ogImage = self.ogTransforms(image)
        downscaledImage = self.downscaleImage(image)

        return downscaledImage, ogImage
    
def getDataloader(path:str, scale:int=4, resolution:int=512, assumeResolution:bool=True, numWorkers:int=4, batchSize:int=64):
    dataset = SuperResolutionDataset(path, scale, resolution, assumeResolution)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    return dataloader