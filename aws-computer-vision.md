# AWS and Computer Vision

__GluonCV__
* GluonCV (runs on apache mxnet engine): a deep learning toolkit for computer vision
    * image classification
    * object detection
    * semantic / instance segmentation
    * pose estimation
* created and maintained by AWS
* Java, Maven, Linux, CPU: 
    * https://mxnet.apache.org/get_started/java_setup.html
    * https://gluon-cv.mxnet.io/install.html

__computer vision tasks:__
* object detection models - with YOLO (Real-Time Object Detection)
* semantic / instance segmentation
* pose estimation

__sources__
* cloudera: AWS Computer Vision: Getting Started with GluonCV
* mxnet support ONNX: https://onnx.ai/supported-tools.html

## Amazon ML Stack
* AI Services: contains high level API's for vision, speech, language, ...
* ML Services: contains Amazon SageMaker
* ML Frameworks & Infrastructure: contains DL frameworks (tensorflow, mxnet, ...)

## Amazon Rekognition
* for image and video analysis
* provides simple API for usage

## Amazon SageMaker
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

## AWS Deep Learning AMI
* Deep Learning Amazon Machine Images: DLAMI
* AMI is a template to create a virtual machine (instance) in EC2 (Amazon Elastic Compute Cloud)
* an AMI includes the OS and any additional software / dependency
* its like purchasing an computer. it has an OS and additional programs
* the DLAMI provides different OS (Ubuntu, Amazon Linux, Windows), preinstalled DL frameworks (mxnet, tensorflow, ...) and Nvidia CUDA drivers
* if you create an DLAMI instance with EC2 then you can access it with SSH (private/public key). you access the instance with the public DNS: `ssh -i "private-key.pem" ubuntu@ec2-...-compute-amazonaws.com`. do not forget to shutdown the instance (for cost reasons). and also delete the instance because you'll get charged for the storage.

## AWS Deep Learning Containers
* these containers provide just another way to set up a _deep learning environment_ on AWS with optimized, prepackaged, container images
* amazon provides a docker container repository: Amazon Elastic Container Registry (ECR)
* Amazon Elastic Container Service: Amazon ECS. ECS do the container orchestration.
* So what are Deep Learning Containers? _These are Docker container images pre-installed with deep learning frameworks_ 
* you can deploy container on:
    * ECS
    * Amazon Elastic Kubernetes Services: Amazon EKS
    * EC2 with DLAMI: connect to instance (with SSH) then docker run
    * on your own machine: Docker and Amazon CLI has to be installed