from mxnet import gluon, nd, init, autograd, metric

from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

import matplotlib.pyplot as plt
from time import time
import numpy as np

# https://stackoverflow.com/a/46616645
def plot_images_demo():
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

print('''
# Neural Network Training & Evaluation
################################################################################

objective: bring all components (autograd, trainer, dataset, and dataloader)
together, to train a network.

in gluon this is done by a training loop.
''')

# get dataset: MNIST fashion
# INFO: dataset will be downloaded once to local mxnet cache ~/.mxnet ...
mnist_train = datasets.FashionMNIST(train=True)
X, y = mnist_train[0]

# X is the image
# y is the class index / label as scalar uint32

print(f'X shape: {X.shape}; dtype: {X.dtype}')
print(f'number of images: {len(mnist_train)}')
# X shape: (28, 28, 1); dtype: <class 'numpy.uint8'>
# number of images: 60000

X, y = mnist_train[0:6]

## transform
# we need to transform dataset to tensor format: channel first, float32
# and normalize with mean 0.13 and standard deviation 0.31
transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)
    ]
)
mnist_train = mnist_train.transform_first(transformer)

## data load
batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

# train_data: batches of images and label pairs
for data, label in train_data:
    print(data.shape, label.shape)
    break
# (256, 1, 28, 28) (256,)

## define model (LeNet architecture) for training
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10),
    )

# init with Xavier (popular for CNN)
net.initialize(init=init.Xavier())

## Loss function
# we define this beside the neural network that is minimized during training
# we use softmax cross entropy

# what does this function?
# it performs softmax on the output to obtain the predicted probability and
# then compares the label to cross entropy
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

## Metric
# measure the network performance as we train
train_acc = metric.Accuracy()

## optimization
trainer = gluon.Trainer(
    net.collect_params(),
    'sgd',
    {
        'learning_rate' : 0.1
    }
)

## training loop
for epoch in range(10):
    train_loss = 0.
    tic = time()
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
    
        trainer.step(batch_size)

        train_loss += loss.mean().asscalar()
        train_acc.update(label, output)
    
    print("epoch[%d] loss:%.3f acc:%.3f perf:%.1f img/sec"%(
        epoch, train_loss / len(train_data),
        train_acc.get()[1],
        len(mnist_train) / (time()-tic)))

# epoch[0] loss:0.841 acc:0.697 perf:4340.6 img/sec
# epoch[1] loss:0.469 acc:0.761 perf:4436.8 img/sec
# epoch[2] loss:0.403 acc:0.790 perf:4337.6 img/sec
# epoch[3] loss:0.368 acc:0.809 perf:4304.3 img/sec
# epoch[4] loss:0.341 acc:0.822 perf:4134.3 img/sec
# epoch[5] loss:0.320 acc:0.832 perf:3844.5 img/sec
# epoch[6] loss:0.305 acc:0.839 perf:4053.4 img/sec
# epoch[7] loss:0.293 acc:0.846 perf:4408.9 img/sec
# epoch[8] loss:0.283 acc:0.851 perf:4517.1 img/sec
# epoch[9] loss:0.273 acc:0.856 perf:4487.0 img/sec

# can be saved also as JSON
net.save_parameters('trained_net.params')
