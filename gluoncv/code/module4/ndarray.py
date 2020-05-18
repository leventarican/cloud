from mxnet import nd
import mxnet as mx 
import numpy as np 

# create a two tier array (= matrix)
matrix = nd.array(((1,2,3), (5,6,7)))
print(matrix)
# [[1. 2. 3.]
#  [5. 6. 7.]]
# <NDArray 2x3 @cpu(0)>

# create same shape with a function 
matrix = nd.ones(shape=(2,3))
print(matrix)
# [[1. 1. 1.]
#  [1. 1. 1.]]
# <NDArray 2x3 @cpu(0)>

# create randomly sampled matrix
matrix = nd.random.uniform(low=-1, high=1, shape=(2,3))
print(matrix)
# [[0.09762704 0.18568921 0.43037868]
#  [0.6885315  0.20552671 0.71589124]]
# <NDArray 2x3 @cpu(0)>

# shape, size and the type (float32, ...)
# size: a matrix is stored linear in memory. here the size is 6
print(matrix.shape)
print(matrix.size)
print(matrix.dtype)

# context: where the data of an array is stored: cpu, gpu, ...?
print(matrix.context)
# cpu(0)
# 0 for the first cpu. if you have 2 gpu's then the second is gpu(1)
# this make mxnet faster then numpy. mxnet uses a optimized library: Intel MKL for CPU operation.
# with gpu its 30x faster then numpy

# convert mxnet ndarray to numpy ndarray
# this operation is synchronous (= further computation is blocked until function is finished.)
matrix = matrix.asnumpy()
print(matrix)

# convert numpy to mxnet ndarray
matrix = nd.array(matrix)
print(matrix)

# calculate the average intensity for every pixel
matrix = nd.random.randint(low=0, high=255, shape=(1, 3, 3))
print(matrix)
flat = nd.flatten(matrix)
flat = flat[0]
print(flat.shape)

a = np.array([[0,1,2,3],[4,5,6,7]])
print(a.shape)
for row in a:
    for col in row:
        print(col)

a = np.arange(8).reshape(2,4)
print(a.shape)

# https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/training/normalization/index.html#Data-Normalization

print('''
get average intensity from image
''')

PIXEL = 28
matrix = nd.random.randint(low=0, high=256, shape=(3, PIXEL, PIXEL))
print(matrix.shape)
M = mx.ndarray.mean(matrix, axis=(0), keepdims=True)
print(M.shape)

print('''
pixelwise normalized image
''')

def get_channel_average_from_batch(batch):
    # print(len(batch[0]))
    M = mx.ndarray.mean(batch[0], axis=(1, 2))
    # print(M)
    return M

test_averages = mx.nd.array([1,2,3,4])
print(test_averages.shape)
test_input = mx.nd.reshape(test_averages, shape=(1,4,1,1)) * mx.nd.ones(shape=(10,4,25,25))
print(test_input.shape)
test_channel_average = get_channel_average_from_batch(test_input)

print(test_averages.asnumpy())
print(test_channel_average.asnumpy())

np.testing.assert_array_almost_equal(test_averages.asnumpy(), test_channel_average.asnumpy())
