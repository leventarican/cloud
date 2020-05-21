from mxnet import gluon, nd, metric

from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

import matplotlib.pyplot as plt

# build model
################################################################################
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

# load the parameters
net.load_parameters('trained_net.params')

# predict
################################################################################
# validation dataset
mnist_valid = datasets.FashionMNIST(train=False)

# data transformation
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)
])

# predict first 6 images
preds = []
for index in range(6):
    image, label = mnist_valid[index]
    image = transform_fn(image).expand_dims(axis=0)
    pred = net(image).argmax(axis=1)
    preds.append(pred.astype('int32').asscalar())

# validation DataLoader
batch_size = 256
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transform_fn), batch_size = batch_size, num_workers=4
)

# accuracy
valid_acc = metric.Accuracy()

# validation loop
for data, label in valid_data:
    output = net(data)
    valid_acc.update(label, output)

print("validation acc: %.3f"%(valid_acc.get()[1]))
# validation acc: 0.895
