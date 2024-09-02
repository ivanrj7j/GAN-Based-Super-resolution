training = True
# specifiies if the model should be trained or not 

trainDatasetPath = "dataset\\train\\high_res"
valDatasetPath = "dataset\\val\\high_res"
# dataset paths 

batchSize = 8
epochs = 100
lr = 1e-4
# training parameters 

shouldLoadMidas = False 
# determines if model should be trained using depth provided by midas 
midasName = "intel-isl/MiDaS"
midasType = "DPT_Hybrid"
# midas configuration 

generatorPath = ""
discriminatorPath = ""
# paths to existing trained generator and discriminator models, if any. weights would be loaded for training.

checkpointInterval = 5
savePath = "checkpoints"
# saves model weights every 'checkpointInterval' epochs

scale = 4
# upscale factor 

device = "cuda"

resolution = 512
assumeResolution = True
numWorkers = 4
# data loader params 