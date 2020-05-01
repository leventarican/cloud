# Data Engineering with Google Cloud Professional Certificate
* Course / Module 1: Google Cloud Platform Big Data and Machine Learning Fundamentals
* Course / Module 2: Modernizing Data Lakes and Data Warehouses with GCP
* Course / Module 3: Building Batch Data Pipelines on GCP
* Course / Module 4: Building Resilient Streaming Analytics Systems on GCP
* Course / Module 5: Smart Analytics, Machine Learning, and AI on GCP
* Course / Module 6: Preparing for the Google Cloud Professional Data Engineer Exam

## Google Cloud Platform Big Data and Machine Learning Fundamentals

### Intro
> _McKinsey research, by 2020, we'll have 50 billion devices connected in the Internet of Things._
> _... only about __one percent__ of the data
generated __today__ is actually
analyzed ..._
* Big Data challenges:
    * migrate existing data workloads (ex. hadoop, spark jobs)
    * analyzing large dataset at scale
    * building __streaming data pipelines__
    * applying machine learning to your data

### Intro to GCP
* GCP was initially build to power google own apps
* GCP infrastructure (building blocks):
    * compute
    * storage
    * network
    * security
* Big Data and ML Products build upon the GCP infrastructure in order to abstract the bare metal way.
> _cloud computing differs from desktop computing: ex. compute and storage are independent._

### Compute power for ML workloads
* ML Model is used for image, video stabilization in google image, youtube, ...
* the pre-trained models / AI building blocks are offered:
    * sight: cloud vision, cloud video intelligence, AutoML vision
    * language: cloud translation, AutoML translation, ...
    * conversion: cloud text-to-speech, ...
* google designed hardware for ML: 
    * TPU: tensor processing unit
    * ASIC: AI accelerator application-specific integrated circuit
    * TPU is an ASIC

### Example: create VM and Storage bucket
* create a VM instance
* create a global unique bucket
* open shell in VM
* access list files in bucket: `gutil ls gs://...`
* copy file to bucket: `gutil cp file gs://...`
* create public link to bucket files

### Data pipelines
* build data pipelines before building ML models from that data
* data pipeline: bring the data to your system

### GCP hierarchy
* _resources_: 
    * BigQuery dataset
    * Cloud storage bucket
    * Compute engine instance
* _projects_:
    * dev
    * test project
    * production
* _folders_ - collections of projects: 
    * team a
        * product 1, product 2
    * team b
* _organization_ - root node of the entire GCP hierarchy
    * optional
    * apply policies (IAM, user access)
> zones and regions physically orginazes resources.
> projects organizes logically resources.

### Networking
* private network
* google layed fiber optic cable that crosses oceans
* the data centers are interconnected. cable diameter ~10cm
* 1 Petabit/sec bandwith
> Google's Jupiter Network can deliver enough bandwidth to
allow 100,000 machines to communicate amongst each other.
> Google's Network, interconnects with the public Internet at more than 90 internet exchanges and more than 100 points of presence worldwide. 
> Google responds to the user's request from an Edge network location that will provide the lowest delay or latency. 

### Security
* Responsibility management
    * On-premise: you manage the the responsibilities
        * hardware, network, OS, identity, web app security, development, usage, access policy, ...
    * IaaS: identity, web app security, development, usage, access policy, ...
    * PaaS: web app security, development, usage, access policy, ...
    * Managed services: usage, access policy
* stored data is encrypted. ex. in BigQuery

### Evolution of data processing frameworks
* know how the frameworks have evolved
* as data growth the needs for handling data at google also growth. Innovation by google:
    * 2002: GFS invented foundation for storage, bigquery
    * 2004: __MapReduce__ paper introduced large scale of data processing
    * hadoop was created
    * 2006: __Bigtable__, inspiration for __Apache HBase__, __MongoDB__
    * 2008: __Dremel__, new approach for data processing
    * 2010 - 2018: Colossus, Flume, Megastore, Spanner, Pub/Sub (for messasing), Tensorflow (for ML), TPU
* these innovations are now provided as services in GCP:
    * 2002: Cloud storage
    * 2004: Dataproc
    * 2006: Bigtable
    * 2008: BigQuery
    * 2010: Dataflow
    * 2011: Datastore
    * 2014: Pub/Sub
    * 2015: ML Engine
    * 2016: Cloud Spanner
    * 2018: AutoML
