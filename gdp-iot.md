# Google Cloud - IoT

* Term _Internet of Things_ was created 1999
* IoT devices in 2030: ~125 billion
* collect data from physical world (information offline) with devices.

## market sectors

* Smart cities - a city that uses technology to improve efficiency, sustainability, and quality of life for people living and working in the city. 
* Industrial IoT - generate value from sensor data. With machine learning and big data
* Connected health - using consumer technologies to connect patients and healthcare providers outside of the hospital. 
* Smart homes - using smart devices to control the environment in a home.

## IoT network

* __device__: interacts with the environment
    * converts information from the physical world in digital data
    * real world data is collected using __sensors__
* __gateway__: gather the data (from device) and communicate with cloud
    * can be a cell phone, microprocessor platform, personal assistant (alexa, google home, ...)
    * does real-time analytics or machine learning (= __edge computing__)
* __cloud__: store, process, analyze data

## Machine Learning on the edge

* run machine learning on device with tensorflow and TPU board
* __predictive maintainance__: predict downtime, detect anomalies, track device status, state, location

## Challenges

* IoT network need throughput. there is no room for bottlenecks between information exchange. data analysis, machine learning and data gathering is done on devices.
* legacy infrastructures need to integrate new technologies
* security: each node is a potential opening for hacking
* dealing with different protocols

## IoT Architecure / Network

* Google's IoT architecture has four stages: data gathering, data ingest, data processing, and data analysis.
* on __data gathering__ stage, edge device gets data from environment and do realtime analitics and ML. _before_ the data is sent to cloud. Doing ML on the edge decreases latency
* __ingest__: with pup/sub data is uploaded to cloud
* __processing__: cleaning, transformation, storing of data
* __analysis__: gain insights, bring ML to production with ai platform

## Security

* devices needs to securely connected to the IoT Network
* Google Cloud IoT do the authentication to the cloud, authorization to pub/sub

## Data types

* sending temperatures is called telemetry. this is read-only
* there is also a second type device state
* you can also send commands from the cloud the control the device. e.g. stop sending data.

## Protocol

* choose HTML, MQTT or both
* MQTT is an industry-standard IoT protocol
* _MQTT is considered to be data focused, while HTTP is document focused. Which means MQTT is better suited to the rigors of IoT._

## Pub/Sub service

* called Cloud pup/sub
* decopling publisher and subscriber
* topics hold messages. a subscription take receives messages from a topic. publisher send messages to a topic

## Cloud IoT Core

* ingest stage
* uses MQTT protocol with single global endpoint: `mqtt.googleapis.com`
* you create a device registry, add a topic to it and add devices to this 

## Cloud Storage

* a unified object storage
* store and retrieve data from any where in the world
* data is stored in a _bucket_. a bucket is defined by:
    * __globally-unique__ name
    * a __geographic location__ where the bucket is stored
    * storage classes: Multi-Regional, Regional, Nearline, and Coldline.

## Dataflow
* _for transforming and enriching data in stream (real time) or batch (historical) modes._
* _Dataflow pipelines are either batch (processing bounded input like a file or database table) or streaming (processing unbounded input from a source like Cloud Pub/Sub)_
* pipelines can be creates also with Apache Beam SDK
    * Apache Kafka, Avro

## Google Cloud Platform (GCP)
* Cloud IoT Core
* Cloud Pub/Sub
* Cloud Storage
* Cloud BigQuery
* Cloud Dataflow (Dataprep Flow, 3rd party: Trifacta)
* Data Studio (not Google Cloud)

### Summary

__ingest__

* IoT Core: create a device registry, add a device
    * use a MQTT Application which simulate the data
* Cloud Pub/Sub: create a topic and add a subscription

__process__

* Cloud Dataflow: create a pipeline to transfer data to storage or bigquery
* Cloud Storage: stream data to bucket

__analyze__

* Cloud BigQuery: create dataset and table
* Cloud Dataprep: manipulate the data
* Data Studio: create a report
* Cloud ML

### Sources

* coursera: Industrial IoT on Google Cloud Platform
* https://cdn.ihs.com/www/pdf/IoT_ebook.pdf
* https://www.youtube.com/watch?v=51bq_Yhuof4 - Google Cloud IoT Solutions
* https://www.youtube.com/watch?v=WAp6FHbhYCk - AWS IoT Services
* https://medium.com/@aallan/hands-on-with-the-coral-usb-accelerator-a37fcb323553 - USB Accelerator, Edge TPU
* https://cloud.google.com/edge-tpu
* https://cloud.google.com/ai-platform - analysis stage: bring ML to production
* https://datastudio.google.com/ - Data Studio, based on Google Drive
* https://github.com/GoogleCloudPlatform/training-data-analyst.git - simulating device
* https://github.com/cagamboa123/training-data-analyst.git  - simulating device

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
