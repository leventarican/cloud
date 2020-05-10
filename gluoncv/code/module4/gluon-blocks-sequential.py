import mxnet as mx 

from mxnet import nd
from mxnet.gluon import nn

# Sequential class: provides function to create sequential models
# this give us a way to stack layers
net = nn.Sequential()

# LeNet architecture: compose a simple
################################################################################
net.add(
    # 6 channels by 5x5 kernel
    nn.Conv2D(6, (5,5), activation='tanh'),
    # pooling siz 2 by 2
    nn.MaxPool2D((2,2)),
    nn.Conv2D(16, (5,5), activation='tanh'),
    nn.MaxPool2D((2,2)),
    # three fully connected layer
    # 120 input units
    nn.Dense(120, activation='tanh'),
    # 84 hidden units
    nn.Dense(84),
    # 10 output units
    nn.Dense(10)
)
print(net)
# Sequential(
#   (0): Conv2D(None -> 6, kernel_size=(5, 5), stride=(1, 1), Activation(tanh))
#   (1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#   (2): Conv2D(None -> 16, kernel_size=(5, 5), stride=(1, 1), Activation(tanh))
#   (3): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
#   (4): Dense(None -> 120, Activation(tanh))
#   (5): Dense(None -> 84, linear)
#   (6): Dense(None -> 10, linear)
# )

net.initialize()

net(nd.ones((1,1,28,28)))
net.forward(nd.ones((1,1,28,28)))

# VGG16 architecture
################################################################################
# good for image classification
# data flow is here more complicated

