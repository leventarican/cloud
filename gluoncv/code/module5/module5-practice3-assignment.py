#!/usr/bin/env python
# coding: utf-8

# # Graded Assessment
# 
# In this assessment you will write a full end-to-end training process using gluon and MXNet. We will train the LeNet-5 classifier network on the MNIST dataset. The network will be defined for you but you have to fill in code to prepare the dataset, train the network, and evaluate it's performance on a held out dataset.

# In[15]:


from pathlib import Path
from mxnet import gluon, metric, autograd, init, nd
import os


# In[16]:


M5_DATA = Path(os.getenv('DATA_DIR', '../../data'), 'module_5')
M5_IMAGES = Path(M5_DATA, 'images')


# ---
# ## Question 1
# 
# ### Prepare and the data and construct the dataloader
# 
# * First, get the MNIST dataset from `gluon.data.vision.datasets`. Use
# * Don't forget the ToTensor and normalize Transformations. Use `0.13` and `0.31` as the mean and standard deviation respectively
# * Construct the dataloader with the batch size provide. Ensure that the train_dataloader is shuffled.
# 
# <font color='red'>**CAUTION!**</font>: Although the notebook interface has internet connectivity, the **autograders are not permitted to access the internet**. We have already downloaded the correct models and data for you to use so you don't need access to the internet. Set the `root` parameter to `M5_IMAGES` when using a preset dataset. Usually, in the real world, you have internet access, so setting the `root` parameter isn't required (and it's set to `~/.mxnet` by default).

# In[19]:


import os
from pathlib import Path

# we need this for dataset and transform ;)
from mxnet.gluon.data.vision import datasets, transforms

print('ready.')

def get_mnist_data(batch=128):
    """
    Should construct a dataloader with the MNIST Dataset with the necessary transforms applied.
    
    :param batch: batch size for the DataLoader.
    :type batch: int
    
    :return: a tuple of the training and validation DataLoaders
    :rtype: (gluon.data.DataLoader, gluon.data.DataLoader)
    """
        
    # YOUR CODE HERE
    #raise NotImplementedError()
    
    t = gluon.data.vision.datasets.MNIST(train=True, root=M5_IMAGES)
    v = gluon.data.vision.datasets.MNIST(train=False, root=M5_IMAGES)
    
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)
    ])
    
    mnist_train = t.transform_first(transformer)
    mnist_validation = v.transform_first(transformer)
    
    train_dataloader = gluon.data.DataLoader(
        mnist_train,
        batch_size=batch,
        shuffle=True,
        num_workers=4
    )
    
    validation_dataloader = gluon.data.DataLoader(
        mnist_validation,
        batch_size=batch,
        shuffle=False,
        num_workers=4
    )
    
    return train_dataloader, validation_dataloader


# In[20]:


t, v = get_mnist_data()
assert isinstance(t, gluon.data.DataLoader)
assert isinstance(v, gluon.data.DataLoader)

d, l = next(iter(t))
assert d.shape == (128, 1, 28, 28) #check Channel First and Batch Size
assert l.shape == (128,)

assert nd.max(d).asscalar() <= 2.9 # check for normalization
assert nd.min(d).asscalar() >= -0.5 # check for normalization


# ---
# 
# ## Question 2
# 
# ### Write the training loop
# 
# * Create the loss function. This should be a loss function suitable for multi-class classification.
# * Create the metric accumulator. This should the compute and store the accuracy of the model during training
# * Create the trainer with the `adam` optimizer and learning rate of `0.002`
# * Write the training loop

# In[24]:


def train(network, training_dataloader, batch_size, epochs):
    """
    Should take an initialized network and train that network using data from the data loader.
    
    :param network: initialized gluon network to be trained
    :type network: gluon.Block
    
    :param training_dataloader: the training DataLoader provides batches for data for every iteration
    :type training_dataloader: gluon.data.DataLoader
    
    :param batch_size: batch size for the DataLoader.
    :type batch_size: int
    
    :param epochs: number of epochs to train the DataLoader
    :type epochs: int
    
    :return: tuple of trained network and the final training accuracy
    :rtype: (gluon.Block, float)
    """
    
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    # we have: 
    # network, 
    # training_dataloader, 
    # batch_size, 
    # epochs
    
    # create:
    # loss function
    # metric accumulator
    # trainer (adam)
    # training loop
    
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_acc = metric.Accuracy()
    trainer = gluon.Trainer(network.collect_params(), 'adam', { 'learning_rate' : 0.002 })
    
    for epoch in range(epochs):
        for data, label in training_dataloader:
            with autograd.record():
                output = network(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_acc.update(label, output)
        training_accuracy = train_acc.get()[1]
        print(training_accuracy)
    return network, training_accuracy


# Let's define and initialize a network to test the train function.

# In[25]:


net = gluon.nn.Sequential()
net.add(gluon.nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=2, strides=2),
        gluon.nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=2, strides=2),
        gluon.nn.Flatten(),
        gluon.nn.Dense(120, activation="relu"),
        gluon.nn.Dense(84, activation="relu"),
        gluon.nn.Dense(10))
net.initialize(init=init.Xavier())


# In[26]:


n, ta = train(net, t, 128, 5)
assert ta >= .95

d, l = next(iter(v))
p = (n(d).argmax(axis=1))
assert (p.asnumpy() == l.asnumpy()).sum()/128.0 > .95


# ---
# ## Question 3
# 
# ### Write the validation loop
# 
# * Create the metric accumulator. This should the compute and store the accuracy of the model on the validation set
# * Write the validation loop

# In[29]:


def validate(network, validation_dataloader):
    """
    Should compute the accuracy of the network on the validation set.
    
    :param network: initialized gluon network to be trained
    :type network: gluon.Block
    
    :param validation_dataloader: the training DataLoader provides batches for data for every iteration
    :type validation_dataloader: gluon.data.DataLoader
    
    :return: validation accuracy
    :rtype: float
    """
    
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    validation_accuracy = metric.Accuracy()
    
    # validation loop
    for data, label in validation_dataloader:
        output = network(data)
        validation_accuracy.update(label, output)
    
    validation_accuracy = validation_accuracy.get()[1]
    
    return validation_accuracy


# In[30]:


assert validate(n, v) > .95


# In[ ]:




