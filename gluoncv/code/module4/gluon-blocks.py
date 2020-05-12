import mxnet as mx 

from mxnet import nd
from mxnet.gluon import nn

import matplotlib.pyplot as plt 

# this code show three building blocks: 3D convolution, max pooling and dense layer
################################################################################

# building block: 2D convolution
################################################################################
# create a  conditional layer
# activation function: ReLU or Rectified Linear Units 
layer = nn.Conv2D(
    channels=1, # how many output channels this network should have. in this case only 1 kernel applied.
    kernel_size=(3,3),
    in_channels=1,  # this input channel is optional because it can be inferred automatically
    strides=(1,1),
    padding=(0,0),
    activation='relu',
    prefix='conv_'
)
print(layer)
# Conv2D(1 -> 1, kernel_size=(3, 3), stride=(1, 1), Activation(relu))

# sobel vertical
layer.initialize(mx.init.Constant([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]))

image = mx.image.imread('../module3/dog.jpg', 0).astype('float32')
image_channel_first = image.transpose((2,0,1))
image_batchified = image_channel_first.expand_dims(axis=0)
output = layer(image_batchified)

# plt.imshow(output.squeeze().asnumpy(), cmap='gray')
# plt.show()

# building block: max pooling
################################################################################
layer = nn.MaxPool2D(pool_size=(4,4))
print(layer)
# MaxPool2D(size=(4, 4), stride=(4, 4), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)

pooled = layer(output)

plt.imshow(output.squeeze().asnumpy(), cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(pooled.squeeze().asnumpy(), cmap='gray')
plt.axis('off')
plt.show()

# building block: dense layer / fully connected layer
################################################################################
# also called: dense layer
# with 3 inputs and 10 outputs
layer = nn.Dense(
    units=10,
    in_units=3,
    activation='relu'
)
print(layer)
# Dense(3 -> 10, Activation(relu))

layer.initialize(mx.init.One())

inputs = mx.nd.ones((1,3))
print(inputs, layer(inputs))
# [[1. 1. 1.]]
# <NDArray 1x3 @cpu(0)> 
# [[3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]]
# <NDArray 1x10 @cpu(0)>

# weights
print( layer.weight.data() )
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]
# <NDArray 10x3 @cpu(0)>

# bias
print( layer.bias.data() )
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# <NDArray 10 @cpu(0)>
