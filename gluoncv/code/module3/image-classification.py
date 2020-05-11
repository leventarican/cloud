import mxnet as mx
import gluoncv as gcv 
import matplotlib.pyplot as plt 

################################################################################
# image classification pipeline
################################################################################

# first transform the image then interprete the neuronal networks output
# for more check: https://github.com/dmlc/gluon-cv/blob/master/docs/tutorials/classification/demo_imagenet.py

# exampe image: mount baker
IMAGE_FILE = 'mt_baker.jpg'

# save image
################################################################################
image_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/mt_baker.jpg'
image_filepath = IMAGE_FILE
gcv.utils.download(url=image_url, path=image_filepath)

# load image as ND array
################################################################################
image = mx.image.imread(IMAGE_FILE)

print('type: ', type(image))
# type:  <class 'mxnet.ndarray.ndarray.NDArray'>
# NDArray is a multidimensional array
# similar to numpy's ndarray

print('shape: ', image.shape)
# shape:  (1458, 3000, 3)
# this array has the HWC layout
# 1. dimension: image height is 1458
# 2. dimension: image width is 3000
# 3. dimension: color channel RGB

print('data type: ', image.dtype)
# data type:  <class 'numpy.uint8'>
# this shows how the values in this array is stored as unsigned integer
# lowest value 0 - highest value - 255

print('minimum value: ', image.min().asscalar())
print('maximum value: ', image.max().asscalar())
# minimum value:  0
# maximum value:  255

# visualize image with pyplot
################################################################################
# convert mxnet array to numpy array for visualizing in pyglot
# plt.imshow(image.asnumpy())
# plt.show()

# transform and batch
################################################################################

# if we give this image now to the network / model we will receive an error
# Check failed: dshp.ndim() == 4U (3 vs. 4) : Input data should be 4D in batch-num_filter-y-x
# the current image is 3D with HWC layout
# the network expect an input with 4 dimensions

# input layout for GluonCV has to be NCHW: N=batch dimension, channel, height, width
# batch dimension?
# the 4th dim is for bunch of images. 
# we stack multiple images along an extra dimension to create a batch of images.
# we do this to improve the network throughput.

# additionally we need a float32 datatype instead of uint8

# and finally we need to normalize the input data. 
# instead of 0-255 range the values average should be (around) 0 with a standard deviation of 1

# we dont need to this by hand. just using the GluonCV transform function
# our model was pre-trained with imagenet1k thus we use the imagenet.transform_eval function
# other datasets have different transform functions
image = gcv.data.transforms.presets.imagenet.transform_eval(image)
# data layout is now NCHW
# height and width is now 224, 224. its re-scaled by the transform process.
# its more manageable (network memory, computation)

print('shape: ', image.shape)
print('data type: ', image.dtype)
print('minimum value: ', image.min().asscalar())
print('maximum value: ', image.max().asscalar())
# shape:  (1, 3, 224, 224)
# data type:  <class 'numpy.float32'>
# minimum value:  -2.117904
# maximum value:  2.2489083

# load model
################################################################################
# load from model zoo
# image is transformed and we can pass the image to the network
network = gcv.model_zoo.get_model('ResNet50_v1d', pretrained=True)
# or: network = gcv.model_zoo.resnet50_v1d(pretrained=true)
# resnet50D is a pre-trained model on imagenet (= image classification dataset)
# INFO - on first call the NN model parameters are downloaded to: ~/.mxnet/models/resnet50_v1d-*.zip
# afterwords its directly loaded from the cache.

# make prediction
################################################################################
# we provide the network our image
# the network will give us a batch of predictions.
# our input was also a batch of images. the transform function did it for us.
# the prediction is a MXNet ndarray
prediction = network(image)
print(prediction.shape)

# our prediction has a shape (1, 1000)
# our input was a batch of images. our output is a batch of predictions.
# therefor remove the extra dimension from prediction
prediction = prediction[0]
# our predtion is now an array of 1000 values: (1000,)
# we used the model that has been pre-trained on imagenet1k, so we have a prediction
# for each of the 1000 classes (car, mountain, cat, ...) from the dataset
# the network have for each class a prediction

print('prediction (raw value / logits) for classes (10 of 1000): ')
print(prediction[990:])
# [ 0.6881457   0.52014035  0.48171854  1.0798138   1.2532476   0.896704
#  -0.433789    0.6877531   0.70875716  0.7740842 ]
# <NDArray 10 @cpu(0)>

# how to interprete these values?
# what we see are raw outputs of the network. also called logits.

# we can convert the logits to probabilities using the softmax function

# logits?
# logits can have value from _negativ infinity_ to _plus infinity_

# how are these probability are calculated using the softmax function?
# the softmax function will give us values form 0 to 1 AND all the values are summed to 1 across the classes
# before normalizing the values are exponentiated: 
# a hight class logit will be a higher class probability. 
# its size depend on the other logits.

# example for softmax function
softmax_example_0 = mx.nd.softmax(mx.nd.array((-0.5, 0.5)))
print('softmax_example_0: ', softmax_example_0)
# [0.26894143 0.7310586 ]
# <NDArray 2 @cpu(0)>
# positive logit leeds to a higher probability then the negative logit: 0.5 > -0.5

softmax_example_1 = mx.nd.softmax(mx.nd.array((-0.5, 1.0)))
print('softmax_example_1: ', softmax_example_1)
# [0.18242551 0.81757444]
# <NDArray 2 @cpu(0)>
# again 1.0 is higher then -0.5. the probability is higher for that AND the sum is 1

# calculate probability
probability = mx.nd.softmax(prediction)
rounded_probability = mx.nd.round(probability*100)/100
print('probability for classes (30 of 1000): ')
print(rounded_probability[970:])
# [0.05 0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.83 0.   0.   0.
#  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
#  0.   0.  ]
# <NDArray 30 @cpu(0)>

# here (output rounded_probability) we see that one class a probability of 83%

# now lets extract the top 5 most likely classes (of 1000).
# we are using the top K function for this
k = 5
topk_indicies = mx.nd.topk(probability, k=k)

# now we need to convert the indicies to human readable labels: 980 to volcano, ...
for i in range(k):
    class_index = topk_indicies[i].astype('int').asscalar()
    class_label = network.classes[class_index]
    class_probability = probability[class_index]
    print('# {} {} ({:0.3}%)'.format(i+1, class_label, class_probability.asscalar()*100))

# 1 volcano (83.2%)
# 2 alp (5.06%)
# 3 valley (0.624%)
# 4 mountain tent (0.537%)
# 5 lakeside (0.496%)
