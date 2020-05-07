import mxnet as mx 
import gluoncv as gcv 
import matplotlib.pyplot as plt 

intro = '''
# image segmentation
image segmentation is a parent term for:
* semantic image segmentation
* instance image segmentation

when use image segmentation?
* to understand the details of an image
* it make low level predictions
* classify every pixel on the image
* image segmentation try to predict the exact boundary of a class
* ex. use for satellite image analysis

image we have an image with trees, house, earth
so we will have three classes: trees, house, earth

## semantic image segmentation
the model will take a pixel and classify the particular 
pixel assigning probabilies to every one of these classes.

ex. the model picks a pixel form the tree: green leaves.
it can be a strong signal for the model to predict this as a class of tree.
so the model may assign following probabilies: house=5%, tree=90%, earth=5%

its common to assign a color for each class: house=blue, tree=yellow, earth=pink

background class is denoted with a class index: -1, color is black

## instance image segmentation
here we segment each instance of an object separatly.
ex. we have two trees. then in instance segmentation each tree is treated separatly.
in other words each tree will have separate pattern.

## GluonCV
GluonCV has a number of semantic segmentation models. 
these models are pre-trained on public available dataset.

## Datasets
ex. dataset contains 20.000 images with 140 classes 

### Pascal VOC (Visual Object Class)
* 2012
* 2900 images
* 6900 object

### COCO (Common Object in Context)
* 2017
* 123.000 images
* 886.000 objects

### ADE20K
* 2016
* 20.000 images
* 400.000 objects

## Models
* semantic segmentation model architecures
    * FCN (with ResNet)
    * PSP (with ResNet)
    * DeepLab (with ResNet)
* more under: https://gluon-cv.mxnet.io/model_zoo/segmentation.html

'''

print(intro)

image_filepath = './dog.jpg'

# load image
################################################################################
# use same image as in object-detection.py
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
transform = '''
we need to create our transformation from scratch.
there is no preset transformation for the dataset or the model we are use.

we can compose multiple transforms into a (mxnet) single transform function: 
* transform #1: ToTensor
    * the data layer will be converted from HWC to CHW
    * data type will be converted from uint8 to float32
* transtorm #2: Normalize
    * normalize the values of the image by using the imagenet1k statistics
'''
print(transform)

from mxnet.gluon.data.vision import transforms
transforms_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, 456, .406], [.229, .224, .225])
])

image = transforms_fn(image)
print('type: ', type(image))
print('shape: ', image.shape)
print('data type: ', image.dtype)
print('minimum value: ', image.min().asscalar())
print('maximum value: ', image.max().asscalar())

batch = '''
create a batch from a single image
add an extra dimension to the beginning of our image
'''
print(batch)
image = image.expand_dims(0)
print(image.shape)


# load model (from zoo)
################################################################################
load = '''
load a pre-trained model from model zoo
we will use the FCN network with the resnet50 backbone which is trained on ADE20K dataset.
network parameters will be cached in ~/.mxnet/models/...
'''
print(load)

network = gcv.model_zoo.get_model('fcn_resnet50_ade', pretrained=True)

# make prediction
################################################################################
prediction = '''
output of network.demo is a mxnet.ndarray
output is a batch of prediction
'''
print(prediction)

output = network.demo(image)
print(output.shape)
# (1, 150, 576, 768)

# we have only one image. lets slice it.
output = output[0]
print(output.shape)
# (150, 576, 768)

# how to interprete it?
# data layout it CHW
# we have 150 channels because ADE20K dataset has 150 classes
# 576 and 768 is corresponding to our image height and width. 
# the values in this array are logits (raw values)

# ex. take a random pixel from the image
px_height, px_width = 300, 500
# layout is CHW. lets slice
px_logit = output[:, px_height, px_width]
# now use softmax function to convert logits to probabilities
px_probability = mx.nd.softmax(px_logit)
px_round_probability = mx.nd.round(px_probability*100)/100
print(px_round_probability)

# use argmax function find the most likely class for the pixel
class_index = mx.nd.argmax(px_logit, axis=0)
# now convert float to int
class_index = class_index[0].astype('int').asscalar()
print(class_index)
# 2

# what is this class index?
# check the lookup table from indecies to labels
from gluoncv.data.ade20k.segmentation import ADE20KSegmentation
class_label = ADE20KSegmentation.CLASSES[class_index]
print(class_label)
# sky

# now perform it for all pixels
# axis=0 corresponds to channel dimension which are the classes
# output is a probability distributed across classes for every pixel
output_proba = mx.nd.softmax(output, axis=0)

output_heatmap = output_proba[2]
# plt.imshow(output_heatmap.asnumpy())
# plt.show()

# visualize most likely class
prediction = mx.nd.argmax(output, 0).asnumpy()
from gluoncv.utils.viz import get_color_pallete
prediction_image = get_color_pallete(prediction, 'ade20k')
plt.imshow(prediction_image)
plt.show()
