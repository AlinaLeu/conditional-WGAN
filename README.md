# conditional-WGAN

This is a modification of https://github.com/lilianweng/unified-gan-tensorflow which again is based on https://github.com/carpedm20/DCGAN-tensorflow.



## Concrete Modifications

(\*) changed code to enable the use of Python 3

(\*) added a few comments

(\*) added an additional model-type, conditional Wasserstein GAN (cWGAN), which outputs MNIST images given specific columns of the image



## Use of cWGAN

To use conditional WGAN choose 'model_type=cWGAN' and specify the columns which shall be used as conditional variables by 'v=[a_0,a_1,a_2,...,a_n]'. Each a_i must be whether of the form [b] or [b_1,b_2], where the first denotes that the column of index b is used and the second that all indices between b_1 and b_2 (exclusive b_2) are used as conditional variable.


### Example 

Run conditional WGAN on MNIST with 10000 iterations and columns with index 2,10 and 11 as conditional variable:

%run main.py --dataset=mnist --model_type=cWGAN --batch_size=64 --input_height=28 --output_height=28 --d_iter=5 --max_iter=10000 --learning_rate=0.00005 --train --v=[[2],[10,12]]