import torch.nn as nn
from utils import loadMidas, loadModels
import config
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import getDataloader, writeSummary, saveModels
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from src import VGGLoss

generator, discriminator = loadModels(config.generatorPath, config.discriminatorPath, None, 3, config.scale, config.device)

if config.shouldLoadMidas:
    midas = loadMidas()
    generator, discriminator = loadModels(config.generatorPath, config.discriminatorPath, midas, 3, config.scale, config.device)

genOpt = Adam(generator.parameters(), config.lr)
discOpt = Adam(discriminator.parameters(), config.lr)

trainLoader = getDataloader(config.trainDatasetPath, config.scale, config.resolution, config.assumeResolution, config.numWorkers, config.batchSize)
valLoader = getDataloader(config.valDatasetPath, config.scale, config.resolution, config.assumeResolution, config.numWorkers, config.batchSize)

writer = SummaryWriter(f"runs/vgg/{round(time.time())}")

bce = nn.BCEWithLogitsLoss()
vggLossFunc = VGGLoss()

genScheduler = ExponentialLR(genOpt, config.decayGamma)
discScheduler = ExponentialLR(discOpt, config.decayGamma)

def trainStep(x, y):
    x, y = x.to(config.device), y.to(config.device)
            
    
    generated = generator(x)

    discReal = discriminator.forward(y)
    discFake = discriminator.forward(generated.detach())
        
    discRealLoss = bce.forward(discReal, torch.ones_like(discReal)-(torch.rand_like(discReal) * 0.08))
    discFakeLoss = bce.forward(discFake, torch.zeros_like(discFake)+(torch.rand_like(discFake) * 0.08))
    discLoss = discFakeLoss + discRealLoss
            
    discOpt.zero_grad()
    discLoss.backward()
    discOpt.step()

    discFake = discriminator.forward(generated)
    adveserialLoss = config.adLambda * bce.forward(discFake, torch.ones_like(discFake))
    vggLoss = config.vggLambda * vggLossFunc.forward(generated, y)
    genLoss = adveserialLoss + vggLoss

    genOpt.zero_grad()
    genLoss.backward()
    genOpt.step()

    return genLoss.item(), discLoss.item()


def train():
    print("Training with VGG Loss...")
    for epoch in range(1, config.epochs+1):
        start = time.time()

        generator.train()
        discriminator.train()

        genLoss = 0
        discLoss = 0

        loop = tqdm(trainLoader, f"[{epoch}/{config.epochs}]", len(trainLoader), leave=False, unit="batch")

        for i, (x, y) in enumerate(loop, start=1):
            currentGenLoss, currentDiscLoss = trainStep(x, y)
            genLoss += currentGenLoss
            discLoss += currentDiscLoss
            loop.set_postfix(genLoss=genLoss/i, discLoss=discLoss/i)
        
        genLoss = genLoss / len(trainLoader)
        discLoss = discLoss / len(trainLoader)

        if epoch % config.decayEvery == 0:
            genScheduler.step()
            discScheduler.step()

        generator.eval()
        discriminator.eval()

        for x, y in valLoader:
            x, y = x.to(config.device), y.to(config.device)
            writeSummary(writer, x, y, generator, {"genLoss": genLoss, "discLoss":discLoss}, epoch)
            break

        if epoch % config.checkpointInterval == 0:
            saveModels(generator, discriminator, config.savePath, f"vgg-{epoch}")
        
        elapsed = round(time.time() - start)
        print(f"[{epoch}/{config.epochs} ] Gen Loss: {float(genLoss):.2f} Dis Loss: {float(discLoss):.2f} ({elapsed}s)")

    saveModels(generator, None, config.savePath, "vgg-final")
    writer.close()
    print("Training completed.")