training = True
# specifiies if the model should be trained or not 

trainDatasetPath = "dataset\\train\\high_res"
valDatasetPath = "dataset\\val\\high_res"
# dataset paths 

batchSize = 3*6
epochs = 250
lr = 1e-4
# training parameters 

shouldLoadMidas = False 
# determines if model should be trained using depth provided by midas 
midasName = "intel-isl/MiDaS"
midasType = "DPT_Hybrid"
# midas configuration 

generatorPath = "checkpoints/vgg/gen-vgg-100.pth"
discriminatorPath = "checkpoints/vgg/dis-vgg-100.pth"
# paths to existing trained generator and discriminator models, if any. weights would be loaded for training.

checkpointInterval = 5
savePath = "checkpoints/vgg"
# saves model weights every 'checkpointInterval' epochs

scale = 4
# upscale factor 

device = "cuda"

resolution = 96
assumeResolution = False
numWorkers = 0
# data loader params 
# numWorkers = 4 meant 269s for vgg net 
# numWorkers = 2 meant 242 for vgg net 

adLambda = 1 #adveserial loss
vggLambda = 6 #vgg loss
# loss weights

useDiscClassifier = False

decayGamma = 0.99
decayEvery = 5