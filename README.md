# GAN-Based-Super-resolution
Implementation of Super resolution models using GANs

## Goals 

- Train a model that can upscale any given image to 4x size
- Implement the [SRGAN](https://arxiv.org/pdf/1609.04802) paper
- Feed the depth data to the model for better understanding. Using [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)
- Only use MSE for comparing
    -  Very bad results, worse that simple bicubic expansion
- Use a hybrid mse first training then, use GAN
- Compare the difference between using GAN, hybrid approach and MSE only.
- Compare results with or without Depth
- Experiment with output of range [0, 1] instead of [-1, 1]
- Instead of using a classifier inside the discriminator itself, try just going with the nxn representation instead that way you can have loss for each region like in pix2pix networks.

`README.md will be updated based on the results of the project later`
