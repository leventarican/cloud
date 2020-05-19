import mxnet as mx 

from mxnet import gluon, nd 
from matplotlib.pyplot import imshow, show

# objective: once we have dataset we need to feed network during training
# gluon provide for this the DataLoader API
# load data from dataset and return mini batches of data
################################################################################

# create dummy gluon dataset (wrap ndarray) and populate data and label 
# with samples from a random uniform distribution

mx.random.seed(10)
X = mx.random.uniform(shape=(10, 3))
y = mx.random.uniform(shape=(10, 1))
dataset = gluon.data.dataset.ArrayDataset(X, y)

# gluon dataloader wraps around the gluon dataset 
# batch_size is the size of the mini batches you want
data_loader = gluon.data.DataLoader(dataset, batch_size=5, last_batch='keep')

for X_batch, y_batch in data_loader:
    print(f'X_batch has shape {X_batch.shape}, and y_batch has shape {y_batch.shape}')
# X_batch has shape (5, 3), and y_batch has shape (5, 1)
# X_batch has shape (5, 3), and y_batch has shape (5, 1)


# load data in parallel using multiprocessing
# set num_workers parameter to the number of CPU's available
################################################################################

# from multiprocessing import cpu_count
# CPU_COUNT = cpu_count()

# data_loader = gluon.data.DataLoader(dataset, batch_size=5, num_workers=CPU_COUNT)

# for X_batch, y_batch in data_loader:
#     print(f'X_batch has shape {X_batch.shape}, and y_batch has shape {y_batch.shape}')


# lazy transformations
################################################################################

from mxnet.gluon.data.vision import transforms

train_dataset = gluon.data.vision.datasets.MNIST(train=True)

# compose a random horizontal flip and a random vertical flip
transform = transforms.Compose(
    [
        transforms.RandomFlipTopBottom(),
        transforms.RandomFlipLeftRight()
    ]
)

# now load the flipped dataset (images) and shuffle it also
train_dataloader = gluon.data.DataLoader(train_dataset.transform_first(transform), 
batch_size=5, shuffle=True)

for X_batch, y_batch in train_dataloader:
    imshow(X_batch[3,:,:,0].asnumpy(), cmap='gray')
    show()
    break
