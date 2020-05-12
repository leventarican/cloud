import mxnet as mx 
from mxnet import nd
from mxnet import metric

# accumulate the metric information
################################################################################
# metrics are indicators for how well a model is performing

# let create a accuracy
accuracy = metric.Accuracy()
labels = nd.array([1,2,3])

## correct prediction
predictions = nd.array([1,2,3])

accuracy.update(labels, predictions)
print( accuracy.get() )
# ('accuracy', 1.0)

## erroneous prediction
accuracy.reset()

predictions = nd.array([1,1,1])
accuracy.update(labels, predictions)

print( accuracy.get() )
# ('accuracy', 0.3333333333333333)
