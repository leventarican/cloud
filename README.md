# cloud

## GluonCV

### setup environment
* example for ubuntu
* in order to work on a clean independent environment we'll use a virtual environment
* then we install mxnet and gluoncv

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
4. install mxnet
``` 
pip install mxnet
``` 
5. install gluoncv
``` 
pip install gluoncv
``` 

### test the setup
* you already activated your environment? then just run `python3 hello.py` 

## IBM Cloud
* free lite account 
* _Lite accounts don't have an expiration date, don't require a credit card and provide limited access to a catalog of more than 40 services_ 
* incl. Watson Visual Recognition
    * Android example: https://heartbeat.fritz.ai/visual-recognition-in-android-using-ibm-watson-9b1fea83e8d
* https://www.ibm.com/cloud/free

## Serverless

* if you manage (configure) instances then its not serverless
* serverless is the discipline to abstract the server
* _The term “Serverless” is confusing since with such applications there are both server hardware and server processes running somewhere_

### Links

* https://martinfowler.com/articles/serverless.html#WhatIsServerless
* https://docs.microsoft.com/en-us/azure/app-service/app-service-web-get-started-java#deploy-the-app - java, maven, azure app service

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

### k8s, docker
1. build container
2. push to container registry
3. kubernetes engine
4. run container