from mxnet import nd, autograd, optimizer, gluon

# demo of gluon trainer with a simple model
net = gluon.nn.Dense(1)
net.initialize()

batch_size = 8
X = nd.random.uniform(shape=(batch_size, 4))
y = nd.random.uniform(shape=(batch_size,))

loss = gluon.loss.L2Loss()

def forward_backward():
    with autograd.record():
        l = loss(net(X), y)
    l.backward()

forward_backward()

# optimizer: stochastic gradient descent
trainer = gluon.Trainer(net.collect_params(), optimizer='sgd', optimizer_params={'learning_rate':1})

# check network parameters
curr_weight = net.weight.data().copy()
print(curr_weight)
# [[ 0.06700657 -0.00369488  0.0418822   0.0421275 ]]
# <NDArray 1x4 @cpu(0)>

# update
trainer.step(batch_size)
print( net.weight.data() )
# [[0.31892323 0.21269077 0.34669656 0.29598683]]
# <NDArray 1x4 @cpu(0)>

print( curr_weight - net.weight.data() * 1 / batch_size )
# [[ 0.02714116 -0.03028122 -0.00145487  0.00512915]]
# <NDArray 1x4 @cpu(0)>

################################################################################
# define an optimzer directly and pass to trainer
# ex. using the AdamOptimizer: a popular adaptive optimizer for deep learning
optim = optimizer.Adam(learning_rate = 1)
trainer = gluon.Trainer(net.collect_params(), optim)

# update network weights
forward_backward()
trainer.step(batch_size)
print( net.weight.data() )
# [[-0.6810826  -0.7873151  -0.65330917 -0.7040191 ]]
# <NDArray 1x4 @cpu(0)>

################################################################################
# changing learning rate

print( trainer.learning_rate )
# 1

trainer.set_learning_rate(0.1)
print( trainer.learning_rate )
# 0.1
