import mxnet as mx 

from mxnet import nd, autograd

x = nd.array(
    [
        [1, 2],
        [3, 4]
    ]
)

print(x)
# [[1. 2.]
#  [3. 4.]]
# <NDArray 2x2 @cpu(0)>

# compute gradient
################################################################################
x.attach_grad()

## define the function f(x) = 2x² 
def f(x):
    return 2 * x**2

with autograd.record():
    y = f(x)

print(y)
# [[ 2.  8.]
#  [18. 32.]]
# <NDArray 2x2 @cpu(0)>

## backward propagation of y
y.backward()

print(x.grad)
# [[ 4.  8.]
#  [12. 16.]]
# <NDArray 2x2 @cpu(0)>

################################################################################

x = nd.array([[1, 0], [0, 1]])
y = nd.array([[1, 1], [1, 1]])

x.attach_grad()
y.attach_grad()

f = lambda x,y : x**2 + y**2
with autograd.record():
    z = f(x, y)
z.backward()

print(x.grad * y.grad)
