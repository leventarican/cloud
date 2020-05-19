import mxnet as mx 

from mxnet import gluon, nd 
from matplotlib.pyplot import imshow, show

# create data
################################################################################
mx.random.seed(42)

# create data with 10 x 3 dimensional data points for X
X = nd.random.uniform(shape=(10, 3))
# create data with 10 x 1 dimensional data points for X
y = nd.random.uniform(shape=(10, 1))

# create gluon data set
################################################################################
# gluon data set wraps out the data
dataset = gluon.data.dataset.ArrayDataset(X, y)

# a tuple of sample data and its corresponding data
sample = dataset[4]
print(sample)
# (
# [0.74707687 0.37641123 0.46362457]
# <NDArray 3 @cpu(0)>, 
# [0.35440788]
# <NDArray 1 @cpu(0)>)

# preloaded data sets in gluon: MNIST and CIFAR10
# CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
################################################################################
# on first call it will be downloaded to mxnet cache folder
train_dataset = gluon.data.vision.datasets.MNIST(train=True)
# valid or test data set
valid_dataset = gluon.data.vision.datasets.MNIST(train=False)

# let visualize it (MNIST data set)
sample_image = train_dataset[19][0]
# print(sample_image)
# imshow(sample_image[:,:,0].asnumpy(), cmap='gray')
# show()

# in ML its common that image data is stored on local machine
################################################################################
# load custom data set with ImageFolderDataset

# image_dataset = gluon.data.vision.datasets.ImageFolderDataset('path')

# define own dataset handling by extending Dataset and DatasetLoader

class CustomDataset(gluon.data.Dataset):
    def __init__(self, dict_data):
        self.dict_data = dict_data
        self.dict_keys = list(dict_data.keys())

    def __getitem__(self, idx):
        return self.dict_data[self.dict_keys[idx]]

    def __len__(self):
        return len(self.dict_data)

dataset = CustomDataset({
    'java' : nd.array(1,),
    'python' : nd.array(2,)
})

print(dataset[1])
# 2.0
# <NDArray  @cpu(0)>
