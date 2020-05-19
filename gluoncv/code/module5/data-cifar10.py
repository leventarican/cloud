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
train_dataset = gluon.data.vision.datasets.CIFAR10(train=True)
test_dataset = gluon.data.vision.datasets.CIFAR10(train=False)

# let visualize it (MNIST data set)
# sample_image = train_dataset[10][0]
# imshow(sample_image[:,:,0].asnumpy())
# show()

print( train_dataset )
