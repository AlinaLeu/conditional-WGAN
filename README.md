# conditional-WGAN

This is a modification of https://github.com/lilianweng/unified-gan-tensorflow which again is based on https://github.com/carpedm20/DCGAN-tensorflow.
I added an additional model-type, conditional Wasserstein GAN (cWGAN), which outputs MNIST images given the 10th column of the image.



## Concrete Modifications

(\*) changed code to enable the use of Python 3
(\*) added a few comments