import mxnet as mx 
import gluoncv as gcv 
import matplotlib.pyplot as plt 

# prepare image for object-detection network
# links:
# https://github.com/dmlc/web-data/tree/master/gluoncv/datasets
# https://gluon-cv.mxnet.io/build/examples_detection/index.html

# download image
################################################################################
image_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/datasets/dog.jpg'
image_filepath = 'dog.jpg'
gcv.utils.download(url=image_url, path=image_filepath)

# load image
################################################################################
image = mx.image.imread(image_filepath)
print('type: ', type(image))
print('shape: ', image.shape)
print('data type: ', image.dtype)
print('minimum value: ', image.min().asscalar())
print('maximum value: ', image.max().asscalar())

# visualize image
################################################################################
# plt.imshow(image.asnumpy())
# plt.show()

# transform and batch
################################################################################
# necessary preprocessing steps for the yolo network is done by the yolo.transform_test function
# height will be resized to 512 pixels
# chw_image (CHW format) is our resized image
# format is NCHW not NHWC. transformed from CHW.
# normalized using the imagenet1k statistics
image, chw_image = gcv.data.transforms.presets.yolo.transform_test(image, short=512)
print('shape: ', image.shape)
print('data type: ', image.dtype)
print('minimum value: ', image.min().asscalar())
print('maximum value: ', image.max().asscalar())

# now to prediction and overlay the detected objects onto the image

# load model (from zoo)
################################################################################
# load pre-trained model from model zoo
# yolo3_darknet53_coco: yolo3 network with a darknet53 backbone trained on coco dataset
# the network parameters (~200MB) are stored in the mxnet cache: ~/.mxnet/models/yolo3_darknet53_coco*.zip
network = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

# make prediction
################################################################################
# result is a tuble not a MXNet ndarray like in image classification
prediction = network(image)
print(type(prediction))

for index, array in enumerate(prediction):
    print('#{} shape: {}'.format(index + 1, array.shape))

#1 shape: (1, 100, 1)
#2 shape: (1, 100, 1)
#3 shape: (1, 100, 4)

# how to interprete the output?
# 1: object class indexes
# 2: object class probabilities
# 3: object bounding box coordinates

# 1: we have 1 image, 100 potential objects and 1 class index per object
# 3: we have 1 image, 100 potential objects and 4 values for each object to define the bounding box

# example: if we have a batch size of 64 and a model that predicts up to 100 objects, 
# the output shapes of the class indicies, probabilities and bounding boxes is
# (64, 100, 1), (64, 100, 1) and (64, 100, 4)

# we have only 1 image. remove other batch dimension.
prediction = [array[0] for array in prediction]

# unpack tuple, give each array its own variable
class_indicies, probabilities, bounding_boxes = prediction

# object class index
####################
# out model can detect potentially 100 object per image. 
# get first 10 objects
k = 10
print(class_indicies[:k])

# [[16.]
#  [ 1.]
#  [ 7.]
#  [ 2.]
#  [13.]
#  [ 0.]
#  [-1.]
#  [-1.]
#  [-1.]
#  [-1.]]
# <NDArray 10x1 @cpu(0)>

# how to interprete this?
# -1 is a special class index for _no object detected_
# so we have 6 detected object in this example: 16, 1, 7, 2, 13, 0
# and 94 undetected objects: 100 - 6
# remember: the indicies are already sorted (top-k)

# now lets gets the label of class index 16. its: dog
class_index = 16
print(network.classes[class_index])

# object probabilities
######################
# how to interprete probabilities?
# in other words its the confidence that the class index is correct
print(probabilities[:k])

# [[ 0.9919528 ]
#  [ 0.9600399 ]
#  [ 0.62269765]
#  [ 0.29241943]
#  [ 0.0179518 ]
#  [ 0.01141728]
#  [-1.        ]
#  [-1.        ]
#  [-1.        ]
#  [-1.        ]]
# <NDArray 10x1 @cpu(0)>

# using a _confidence threshold_ of 50% then in this example 3 objects are detected: 0.9919528, 0.9600399, 0.62269765
# -1 is no _confidence score_

# object boundind boxes
#######################
print(bounding_boxes[:k])

# [[116.53647  201.33238  281.90323  482.09088 ]
#  [ 93.92972  107.739395 504.7513   375.75427 ]
#  [416.78827   69.80066  615.0179   148.89008 ]
#  [416.78827   69.80066  615.0179   148.89008 ]
#  [ 90.883545 125.878815 504.44495  402.7955  ]
#  [532.78235   91.84185  547.3104   104.69111 ]
#  [ -1.        -1.        -1.        -1.      ]
#  [ -1.        -1.        -1.        -1.      ]
#  [ -1.        -1.        -1.        -1.      ]
#  [ -1.        -1.        -1.        -1.      ]]
# <NDArray 10x4 @cpu(0)>

# visualize prediction
################################################################################
# we can use the GluonCV bounding box plot function

# the following objects are detected: dog, bike, car

# the network missed the tree. why? because the model is pre-trained on coco.
# coco doesnt have an object class for tree.
gcv.utils.viz.plot_bbox(chw_image, bounding_boxes, probabilities, class_indicies, class_names=network.classes)
plt.show()
