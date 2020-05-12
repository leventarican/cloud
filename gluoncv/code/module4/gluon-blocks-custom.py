import mxnet as mx 

from mxnet import nd
from mxnet.gluon import nn

# ojective: create more complex data flow
################################################################################

## MLP: Multi Layer Percepton
# https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html
# MLP inherits / extends the Block class
class MLP(nn.Block):
    def __init__(self, hidden_units=256, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(hidden_units, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        y = self.hidden(x)
        return self.output(y)

net = MLP(hidden_units=512)
net.initialize()
net(nd.random.uniform(shape=(1,64)))

## Siamese Network
# good architecture for comparing two images
class SiameseNetwork(nn.Block):
    def __init__(self, hidden_units=256, **kwargs):
        super(SiameseNetwork, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        with self.mlp.name_scope():
            self.mlp.add(
                nn.Dense(128, activation='relu'),
                nn.Dense(128, activation='relu'),
                nn.Dense(128, activation='relu')
            )
        
        def forward(self, x1, x2):
            y1 = self.mlp(x1)
            y2 = self.mlp(x2)
            # re-shape it
            y1 = y1.expand_dims(axis=1) # add dummy dimension
            y2 = y2.expand_dims(axis=2) # Y1: (N, 1, C) Y2: (N, C, 1)
            return nd.batch_dot(y1, y2)

net = SiameseNetwork()
net.initialize()
x1 = nd.random.uniform(shape=(1,64))
x2 = nd.random.uniform(shape=(1,64))
