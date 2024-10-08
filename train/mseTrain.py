import torch.nn as nn
from utils import loadMidas, loadModels
import config
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch import GradScaler, autocast
from utils import getDataloader, writeSummary, saveModels
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

model, _ = loadModels(config.generatorPath, "", None, 3, config.scale, config.device)
if config.shouldLoadMidas:
    midas = loadMidas()
    model, _ = loadModels(config.generatorPath, "", midas, 3, config.scale, config.device)

lossFunc = nn.MSELoss()

optimizer = Adam(model.parameters(), config.lr)

trainLoader = getDataloader(config.trainDatasetPath, config.scale, config.resolution, config.assumeResolution, config.numWorkers, config.batchSize)
valLoader = getDataloader(config.valDatasetPath, config.scale, config.resolution, config.assumeResolution, config.numWorkers, config.batchSize)

writer = SummaryWriter(f"runs/mse/{round(time.time())}")

genScheduler = ExponentialLR(optimizer, config.decayGamma)

def train():
    print("Training...")
    for epoch in range(1, config.epochs+1):
        start = time.time()
        model.train()
        loss = 0
        loop = tqdm(trainLoader, f"[{epoch}/{config.epochs}]", len(trainLoader), leave=False, unit="batch")

        for i, (x, y) in enumerate(loop, start=1):
            x, y = x.to(config.device), y.to(config.device)

            generated = model(x)
            currentLoss = lossFunc(generated, y)
            loss += currentLoss
            
            optimizer.zero_grad()
            currentLoss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item()/i)
        
        loss = loss / len(trainLoader)

        if epoch % config.decayEvery == 0:
            genScheduler.step()
            print(f"Decaying [{optimizer.param_groups[0]["lr"]}]")

        model.eval()

        for x, y in valLoader:
            x, y = x.to(config.device), y.to(config.device)
            writeSummary(writer, x, y, model, {"loss":loss}, epoch)
            break

        if epoch % config.checkpointInterval == 0:
            saveModels(model, None, config.savePath, f"mse-{epoch}")
        
        elapsed = round(time.time() - start)
        print(f"[{epoch}/{config.epochs} ] Loss: {float(loss):.2f} ({elapsed}s)")

    saveModels(model, None, config.savePath, "mse-final")
    writer.close()
    print("Training completed.")