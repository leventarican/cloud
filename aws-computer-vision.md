# AWS and Computer Vision

## GluonCV
* GluonCV: a deep learning toolkit for computer vision
* runs on Apache MXNet engine
* created and maintained by AWS
* Java, Maven, Linux, CPU: 
    * https://mxnet.apache.org/get_started/java_setup.html
    * https://gluon-cv.mxnet.io/install.html
* GluonCV was also created to close the a gap between model experimentation and applicable model deployment
* GluonCV _implements_ models for image classifications, ...
* these models are accessible in __model zoo__.
* these models are pre-trained on public available dataset with millions of images.
* what is a model? 
    * simply defined: a function with an input and an output
    * in CV a model takes an image as input and generate a prediction as output
    * the term _network_ and _model_ is interchangeable
    * for different task you may need different models. it depends on prediction _accuracy_ and _compute resource consumption_
* GluonCV covers following computer vision tasks:
    * image classification
    * object detection models (ex. YOLO: Real-Time Object Detection)
    * semantic / instance segmentation
    * pose estimation

## Apache MXNet
* Apache MXNet: is a deep learning framework
* with MXNet you can build and train neuronal network models 
* it support different languages: Java, Python, R, C++, ...
* MXNet has a lot of pre-trained models in model-zoo
* MXNet supports ONNX: https://onnx.ai/supported-tools.html

## AWS: ML Stack
* AI Services: contains high level API's for vision, speech, language, ...
    * _Amazon Rekognition_
* ML Services
    * _Amazon SageMaker_
* ML Frameworks & Infrastructure: 
    * DL Frameworks TensorFlow, MXNet, Pytorch, ...
    * Infrastructure: EC2, Deep Learning Containers, IoT Greengrass
        * _AWS Deep Learning AMI_

### Amazon Rekognition
* for image and video analysis: object, scene and activity detection
* provides simple API for usage

### Amazon SageMaker
* SageMaker service is compososed of many services: label, build & train, tune, compile, deploy
* amazon sagemaker has a service for labeling for supervised learning. ex. when classification a dog image. on train a human has to label it: 
    * a flag: yes it is a dog
    * pixel coordinates of the dog in the image
* a jupyter notebook with preinstalled (CONDA environments) _apache mxnet, tensorflow, pytorch, chainer_ and non-deeplearning frameworks _scikit-learn and Spark ML_.
* in the jupyter notebook you can write your own ML models, train it, ...
* or you can use build-in algorithms: _K-Means, K-Nearest Neighbors (k-NN), BlazingText_, ...
* further algorithm by 3rd pary is listed in AWS Marketplace
* you can train a model __locally__ on amazon sagemaker notebook or use __model training jobs__ (needed infrastructure / instances is created instantly)
* model is stored is S3
* model optimization jobs: compiles the trained model into exe
* deployment
    * __amazon sagemaker endpoint__: http request
    * __AWS IoT Greengrass__: for deployment on edge devices; see also Amazon SageMaker Neo
* the workflow is controlable by AWS CLI 
* or SDK's in python `import sagemaker`

### AWS Deep Learning AMI
* Deep Learning Amazon Machine Images: DLAMI
* AMI is a template to create a virtual machine (instance) in EC2 (Amazon Elastic Compute Cloud)
* an AMI includes the OS and any additional software / dependency
* its like purchasing an computer. it has an OS and additional programs
* the DLAMI provides different OS (Ubuntu, Amazon Linux, Windows), preinstalled DL frameworks (mxnet, tensorflow, ...) and Nvidia CUDA drivers
* if you create an DLAMI instance with EC2 then you can access it with SSH (private/public key). you access the instance with the public DNS: `ssh -i "private-key.pem" ubuntu@ec2-...-compute-amazonaws.com`. do not forget to shutdown the instance (for cost reasons). and also delete the instance because you'll get charged for the storage.

### AWS Deep Learning Containers
* these containers provide just another way to set up a _deep learning environment_ on AWS with optimized, prepackaged, container images
* amazon provides a docker container repository: Amazon Elastic Container Registry (ECR)
* Amazon Elastic Container Service: Amazon ECS. ECS do the container orchestration.
* So what are Deep Learning Containers? _These are Docker container images pre-installed with deep learning frameworks_ 
* you can deploy container on:
    * ECS
    * Amazon Elastic Kubernetes Services: Amazon EKS
    * EC2 with DLAMI: connect to instance (with SSH) then docker run
    * on your own machine: Docker and Amazon CLI has to be installed

## Module 3: GluonCV Start
* understand how to use pre-trained models for computer vision tasks
* as previously defined you can use GluonCV on local machine (by setup environment) or use pre-installed environment in a Amazon Sagemaker instance, Amazon Elastic Compute Cloud (EC2), Amazon Deep Learning AMI (DLAMI)

### Setup Virtual Environment
Example for ubuntu. in order to work on a clean independent environment we'll use a virtual environment. then we install mxnet and gluoncv

1. create virtual environment: only once
``` 
python3 -m venv gluoncv
``` 
2. activate virtual environment
``` 
cd gluoncv
source bin/activate
``` 
3.  deactivate virtual environment
``` 
deactivate
``` 

### Install MXNet and GluonCV
``` 
pip install mxnet
pip install gluoncv

# for CPU optimized
# pip install mxnet-mkl

# for GPU optimized
# pip install mxnet-cu101
``` 

### Image Classification with a pre-trained model
* objective is to classify the image from a list of predetermined classes
* when making a prediction the model will assign a probability to each of these classes
* ex. we have a mountain image and our classes are: mountain, beach, forest
    * the model will now give probabilities to each class: mountain 80%, beach 0%, forst 20%
* GluonCV Models are pre-trained on public available dataset

#### Datasets
* Datasets for Image Classification
* __CIFAR-10__ (Canadian Institute for Advanced Research) used over 10 years for computer research. it includes 10 basic classes: cars, cats, dogs, ... and 60000 images. its a small dataset with low resolution images (32x32).
* __ImageNet__ is another images classification dataset. released by prinston university in 2009. with 14 mio images and 22.000 classes. refered as _ImageNet22k_. models are pre-trained on a sub-set of it (_Imagenet1k_: 1.000.000 images and 1000 classes).

#### Models
* Neuronal Network Models for image classification
* there are a lot of different models architectures (classification model architectures)
    * __ResNet__ is popular
    * ResNet has different variants: ResNet18, ResNet50
    * __MobileNet__ is good for mobile phones
    * or: VGG, SqueezeNet, DenseNet, AlexNet, DarkNet, Inception, ...
* how to decide which model to take? accurary, memory consumption, ...
* as already wrote, GluonCV implement these model based on the research paper. The GluonCV models are compared to other implementations optimized. The GluonCV version to ResNet-152 is called ResNet-152D.

#### Code Examples
``` 
image-classification.py
``` 

### Object Detection with a pre-trained model
* understand the content of an image
* ex. medical image analysis, self driving cars, ...
* locate object in an image with a box (called bounding box) 
* additionally the located object is classified from a list of predetermined classes
* the model will also give a probability for each class

#### Datasets
* __Pascal VOC__ (Visual Object Class)
    * 2007 version: 10000 images and 24500 object classes
    * 2012 version: 11500 images and 27500 object classes
* __COCO__ (Common Object in Context)
    * 2017
    * 123000 images
    * 886000 objects
    * 80 object classes: Pascal VOC classes are included. Additionally following categories are more covered: sports, food, household objects (table, tv, computer, ...)

#### Models
* Object detection model architectures: 
    * __Faster-RNN__: with ResNet
        * Faster-RNN is an extension of ResNet with additional components
        * the network output need coordinates for bounding box, ...
        * ResNet is called the base network or backbone
    * __SSD__: with VGG or ResNet or MobilNet
    * __YOLO__: with Darknet or MobileNet

#### Code Examples
``` 
object-detection.py
``` 

### Image Segmentation with a pre-trained model
* see code documentation: `image-segmentation.py`

### Neural Network Essentials
* the pre-trained models are based of components called Neural Networks especially __Convolutional Neural Networks__

#### Fully Connected Network
* a fundamental network is called the __fully connected network__
* is called fully connected because all inputs are connected to outputs
* its a general purpose network that makes no assumption about the input data
* the network begins from 0. it has no special information from the image and has to _re-learn_ the relationship between  the pixels

![source: aws training](./gluoncv/fully-connected-network.png)
* example of a fully connected network with 4 inputs with 3 fully connected layers
* the 4 inputs can correspond to 4 pixels (if we have a 2x2 image)
* we have 3 outputs (with predictions) because of 3 classes
* an image is composed of pixels. and each pixel of an RGB image will have three values that encode the intensity of red, green and blue colors.
* more simplified example. we have a gray scale image
    * each pixel represent the intensity: 0 is black and 1 is white
    * now all pixels are flattened for the first layer: pixel to input
* all connections are weighted
    * with this weights the activation function is activated and produces the output
    * with a activation function like __Sigmoid function__ the network has the ability to _learn_
    * therefore the train objective is to find the best weights
* these connection weights are called also __network parameters__
    * real world models have billions of network parameteres or even more
* pre-trained models have already good network parameter that have been learned already
    * we download a file with these values and create a model based on it

#### Convolutional networks
* used in computer vision tasks
* _Convolutional neural networks learn local patterns from small neighborhoods of pixels, unlike fully connected networks that learned patterns across all pixels of the image._
* two most important operations: the convolution operation and the max-pooling operation
* a component of the convolution operation is the __kernel or filter__
* the input goes through this kernel and generate an output to learn pattern or extract features 
* you can achive by this operation: edge feature extraction, or blur image, sharpen
* with deep learning we learn the parameter for the best suited kernel values
* max pooling reduces the dimension by getting the max value of a defined kernel ex. 2x2 (4 input values) the output will be 1 value

__what is a feature?__

* the number of features is equal to number of nodes in the input layer
* if we want to classify man or woman then the attributes (height, hair length, ...) of them are the features
* if we want to classify images (cat / dog) then our features = inputs. the next layer can then do __features extraction__ like cat ears vs dog ears, ...

## Module 4: Gluon Fundamentals
* Understand the mathematics behind NDArray
* Understand when to use different Gluon blocks including convolution, dense layers, and pooling
* Compose Gluon blocks into complete models
* Understand the difference between metrics and loss 

### N-dimensional arrays
* ndarrays also called __tensors__
* vectors and matrices of N-dimensions
* used in deep learning to represent input, output, ...

![source: aws training](./gluoncv/ndarray.png)

### NDArray
* NDArray: MXNet's primary tool for storing and transforming numerical data
* examples: `ndarray.py, ndarray-operations.py`

### Gluon Blocks
* how to create a neuronal network using the gluon API of mxnet
    * examples: `gluon-blocks.py, gluon-blocks-init.py`
* create a sequential block to compose a sequence of layers to create a neural network
     * examples: `gluon-blocks-sequential.py, gluon-blocks-custom.py`
* visualize gluon models (blocks) to understand it better

## Sources, Links
* cloudera: AWS Computer Vision: Getting Started with GluonCV
* a good intro into 2D Convolutions with mxnet: https://medium.com/apache-mxnet/convolutions-explained-with-ms-excel-465d6649831c
* good slides about GluonCV: https://github.com/dmlc/web-data/blob/master/gluoncv/slides/Classification.pdf
