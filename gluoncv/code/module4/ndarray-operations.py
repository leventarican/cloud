from mxnet import nd
import mxnet as mx 
import numpy as np 

# common NDArray operations
################################################################################
x = mx.nd.random.uniform(shape=(2,3))
y = mx.nd.full(shape=(2,3), val=2)
print(x)
print(y)
# [[0.5488135  0.5928446  0.71518934]
#  [0.84426576 0.60276335 0.8579456 ]]
# <NDArray 2x3 @cpu(0)>

# [[2. 2. 2.]
#  [2. 2. 2.]]
# <NDArray 2x3 @cpu(0)>

## multiplication
print( x * y )
# [[1.097627  1.1856892 1.4303787]
#  [1.6885315 1.2055267 1.7158912]]
# <NDArray 2x3 @cpu(0)>

## exponentiation
print( y.exp() )

# when you want to plot it in pyplot you need to convert array to numpy!

## indexing 
# important operation when perform on tensors
# indexing is picking a specific element in matrix: read / write

# ex. 
# get all elements in the 1. dimension (= rows)
# 2. dimension: all elements from 1. index (2. column) to 3. index (3. column; excluded)
print( x[:, 1:3] )
# [[0.5928446  0.71518934]
#  [0.60276335 0.8579456 ]]
# <NDArray 2x2 @cpu(0)>

# writing
# 2. row; 1. and 2. column
y[1:2, 0:2] = 4
print(y)
# [[2. 2. 2.]
#  [4. 4. 2.]]
