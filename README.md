# cloud

## Google Cloud - IoT

* Term _Internet of Things_ was created 1999
* IoT devices in 2030: ~125 billion
* collect data from physical world (information offline) with devices.

__market sectors__

* Smart cities - a city that uses technology to improve efficiency, sustainability, and quality of life for people living and working in the city. 
* Industrial IoT - generate value from sensor data. With machine learning and big data
* Connected health - using consumer technologies to connect patients and healthcare providers outside of the hospital. 
* Smart homes - using smart devices to control the environment in a home.

__IoT network__
* _device_: interacts with the environment
    * converts information from the physical world in digital data
* _gateway_: gather the data (from device) and communicate with cloud
    * can be a cell phone, microprocessor platform, personal assistant (alexa, google home, ...)
    * does real-time analytics or machine learning (= __edge computing__)
* _cloud_: store, process, analyze data

__Machine Learning on the edge__
* run machine learning on device with tensorflow and TPU board
* _predictive maintainance_: predict downtime, detect anomalies, track device status, state, location

IoT Architecture
IoT Vocabulary

__sources__
* coursera: Industrial IoT on Google Cloud Platform
* https://cdn.ihs.com/www/pdf/IoT_ebook.pdf
* https://www.youtube.com/watch?v=51bq_Yhuof4 - Google Cloud IoT Solutions
* https://www.youtube.com/watch?v=WAp6FHbhYCk - AWS IoT Services
* https://medium.com/@aallan/hands-on-with-the-coral-usb-accelerator-a37fcb323553 - USB Accelerator, Edge TPU
* https://cloud.google.com/edge-tpu

## AWS and Computer Vision

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

## cloud data platform
* data lake: _A data lake is a storage repository that holds a vast amount of raw data in its native format until it is needed._
* a data platform has the following layers:
    * ingest
    * storage
    * processing
    * serving
* book source: Designing Cloud Data Platforms, Manning

__processing layer__
* here comes data processing framework in use: apache spark, bean, flink
* with these framework you can: transform, validate, clean data

__serving layer__
* aim is to provide the data to end user


## kubernetes, k8s
* _automating deployment, scaling, and management of containerized applications._
* use https://microk8s.io/ on laptop
* _compatible with Azure AKS, Amazon EKS, Google GKE_

### install microk8s (on ubuntu)
* install with snap: `sudo snap install microk8s --classic`

### start / stop / status k8s (after installing)
 * you need to run command with sudo or add your user to the group _microk8s_. e.g. `sudo microk8s.status`.
* `microk8s.start`
* `microk8s.stop`
* `microk8s.status`
```
sudo microk8s.status
microk8s is running
addons:
dashboard: disabled
dns: disabled
istio: disabled
jaeger: disabled
prometheus: disabled
registry: disabled
storage: disabled
```

### access kubernates
* `microk8s.kubectl get all --all-namespaces`
* `microk8s.kubectl get services`
* `microk8s.kubectl get nodes`

### links
* https://ubuntu.com/tutorials/install-a-local-kubernetes-with-microk8s#1-overview
* https://github.com/ubuntu/microk8s
* https://microk8s.io/docs/
* https://microk8s.io/#get-started

## google cloud platform
* k8s, docker

1. build container
2. push to container registry
3. kubernetes engine
4. run container