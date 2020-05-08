import mxnet as mx 

from mxnet import nd, init
from mxnet.gluon import nn

import matplotlib.pyplot as plt 

################################################################################

# init.Constant
# init.Normal
# init.Zero
# init.One
# ...

## init with Xavier (search for more)
layer = nn.Conv2D(
    channels=1,
    kernel_size=(3,3),
    in_channels=1
)
layer.initialize(init.Xavier())
print( layer.weight.data() )
# [[[[ 0.05636501  0.10720772  0.24847925]
#    [ 0.39752382  0.11866093  0.41332   ]
#    [ 0.05182666  0.4009717  -0.08815584]]]]
# <NDArray 1x1x3x3 @cpu(0)>

## init with ones
# set it directly
layer.weight.set_data(nd.ones((1,1,3,3), ctx=mx.cpu()))
print( layer.weight.data() )
# [[[[1. 1. 1.]
#    [1. 1. 1.]
#    [1. 1. 1.]]]]
# <NDArray 1x1x3x3 @cpu(0)>
