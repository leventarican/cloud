import mxnet as mx 

from mxnet import gluon, nd 
from matplotlib.pyplot import imshow, show

# CIFAR10
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
# with 6000 images per class. There are 50000 training images and 10000 test images. 
#
# Each sample is an image (in 3D NDArray) with shape (32, 32, 3).
# https://www.cs.toronto.edu/~kriz/cifar.html
# https://mxnet.apache.org/api/python/docs/api/gluon/data/vision/datasets/index.html#mxnet.gluon.data.vision.datasets.CIFAR10
################################################################################
# on first call it will be downloaded to mxnet cache folder
train_data = gluon.data.vision.datasets.CIFAR10(train=True)
val_data = gluon.data.vision.datasets.CIFAR10(train=False)

# let visualize it (MNIST data set)
# sample_image = train_data[10][0]
# imshow(sample_image)
# show()

# training batch contains 5000 image for each class
# 10 * 5000 = 50000 training images

# test (validation) batch contains 1000 images for each class
# 10 (classes) * 1000 (images) = 10000 (test images)

# For example, the first class (label) in the CIFAR10 training dataset is 6, 
# this corresponds to index 0 in the training dataset. 
sample_image = train_data[0][1]
print(f'class: {sample_image}')
# class: 6

# example with validation / test batch dataset
val_indices = {}

for i in range(10000):
    val_indices[val_data[i][1]] = i

print(val_indices)

v = val_data
i = val_indices

for k in range(10):
    assert v[i[k]][1] == k
