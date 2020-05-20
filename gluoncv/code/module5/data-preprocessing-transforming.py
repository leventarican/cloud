import mxnet as mx 
from mxnet import gluon, nd 

# ex. transformation is rescaling: 
# change the scale of input pixel in image dataset
# from between 0 and 255 to between 0 and 1
################################################################################

def transform_fn(data, label):
    data = data.astype('float32') / 255
    return data, label

# now pass this function to dataset
train_dataset = gluon.data.vision.datasets.MNIST(train=True).transform(transform_fn)
test_dataset = gluon.data.vision.datasets.MNIST(train=False).transform(transform_fn)

sample_image = train_dataset[19][0]
# max value in sample image. expected 1.0
m = nd.max(sample_image)
print(m)
# [1.]
# <NDArray 1 @cpu(0)>

# but gluon data API has already implemented commonly used 
# transformation functions in computer vision
################################################################################
# gluon.data.vision.transforms

from mxnet.gluon.data.vision import transforms

# ex. ToTensor transformation
# converts an image NDArray of shape HWC in range [0; 255] to 
# float32 tensor NDArray of shape CHW in range [0; 1]
################################################################################

train_dataset = gluon.data.vision.datasets.MNIST(train=True)

print( train_dataset[19][0].shape )
# (28, 28, 1

to_tensor = transforms.ToTensor()
# transform apply to the _first_ entry point and not to the label
train_dataset = train_dataset.transform_first(to_tensor)
print( train_dataset[19][0].shape )
# (1, 28, 28)

# now we have the tensor format: CHW

# another transformation example used in normalization
# normalize a tensor of shape CHW with mean and standard deviation
################################################################################

mean, std = (0.1307,), (0.3081,)
normalize = transforms.Normalize(mean, std)
train_dataset = train_dataset.transform_first(normalize)

# print(train_dataset[19][0])

# compose: composing sequentially multiply transformation and apply them at once
################################################################################
transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)

train_dataset = gluon.data.vision.datasets.MNIST(train=True).transform(transform_fn)

print(type(train_dataset))

# transformation can also be used an data augmentation: resize, centercrop, ...
################################################################################
