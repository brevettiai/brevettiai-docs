# Basic Image Classifier model example
This document introduces a simple image classifier model, to show how to build the necessary packages to host a model on the Brevetti AI platform. It includes the following necessary steps
* Model training code that can also produce a user configuration ```settings_schema.json``` file
  - should accept cmd args with **job_id** and **api_key** or find sagemaker hyperparameter file with the same
  - should produce a model artifact
  - these steps are handled by the ```brevettiai``` *Job* object
* A simple Dockerfile to run the code - this is how the script is embedded on the platform

## Basic keras Image classifier model
As a minimal model example. The code below will serve as a simple model for image classification based on the MobileNet architecture. It accepts any number of classes and the image size may be specified.
For training regularization, it includes a Dropout layer.
The model and training code can be found on the documentation github repository [image_classifier.py](https://github.com/brevettiai/brevettiai-docs/blob/master/src/tutorials/image_classifier_101/image_classifier.py)


```python
import tensorflow as tf

def build_image_classifier(classes: list, image_shape: tuple):
    # Model backbone is the MobileNetV2
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=image_shape, include_top=False, weights="imagenet"
    )
    # Features are pooled and the output layer consists of a single dense layer
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(len(classes), activation='softmax', name="---".join(classes))
    ])
    # Model is compiled with ```categorical_crossentropy``` loss and reports accuracy metric
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

    2022-03-23 21:28:07.155998: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-03-23 21:28:07.156031: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


Using some default image size and classes a model generated with the above code, and we can verify that it seems to produce a valid keras model.


```python
# test run of image classification build code
test_model = build_image_classifier(["good", "bad"], (224, 224, 3))
test_model.summary()
```

    2022-03-23 21:28:08.724157: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-03-23 21:28:08.724190: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2022-03-23 21:28:08.724209: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fv-az96-151): /proc/driver/nvidia/version does not exist
    2022-03-23 21:28:08.724415: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    9412608/9406464 [==============================] - 0s 0us/step
    9420800/9406464 [==============================] - 0s 0us/step
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     mobilenetv2_1.00_224 (Funct  (None, 7, 7, 1280)       2257984   
     ional)                                                          
                                                                     
     global_max_pooling2d (Globa  (None, 1280)             0         
     lMaxPooling2D)                                                  
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     good---bad (Dense)          (None, 2)                 2562      
                                                                     
    =================================================================
    Total params: 2,260,546
    Trainable params: 2,226,434
    Non-trainable params: 34,112
    _________________________________________________________________


    /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/keras/optimizer_v2/adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(Adam, self).__init__(name, **kwargs)


## Model training *Job*
To train and export the model, construct a training object that uses the following ```brevettiai``` components
 - the *Job* class implements the [Job API](https://docs.brevetti.ai/developers/python-sdk-brevettiai/2_brevettiai_job_api_platform_interfaces_documentation) which gives access to the Brevetti AI platform and the dataset ressources.
 - the *Settings* object ```Job.Settings``` specifies the configurable elements that we will use in the script. The parameters of these objects / variables may be specified by the platform user
 - ```package_saved_model``` module saves tensorflow saved_model file as tar gz for the deployment
 
 To install the ```brevettiai``` package simply run ```pip install -U git+https://bitbucket.org/criterionai/core```


```python
pip install brevettiai[tfa]
```

    Requirement already satisfied: brevettiai[tfa] in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (0.3.4)
    Requirement already satisfied: backoff>=1.10 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.11.1)
    Requirement already satisfied: pandas<1.2,>=1.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.1.5)
    Requirement already satisfied: tf2onnx>=1.9.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.9.3)
    Requirement already satisfied: cryptography>=36.0.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (36.0.2)
    Requirement already satisfied: requests>=2.23 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (2.27.1)
    Requirement already satisfied: plotly>=4.14.3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (5.6.0)
    Requirement already satisfied: altair==4.1.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (4.1.0)
    Requirement already satisfied: mmh3>=3.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (3.0.0)
    Requirement already satisfied: pydantic>=1.9.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.9.0)
    Requirement already satisfied: scikit-learn>=0.23.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.0.2)
    Requirement already satisfied: tqdm>=4.62 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (4.63.0)
    Requirement already satisfied: onnxruntime>=1.6.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.10.0)
    Requirement already satisfied: numpy>=1.18.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.21.5)
    Requirement already satisfied: shapely>=1.7.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (1.8.1.post1)
    Requirement already satisfied: configparser>=5.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (5.2.0)
    Requirement already satisfied: minio<7.1,>=7.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (7.0.4)
    Requirement already satisfied: tensorflow_addons>=0.16.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai[tfa]) (0.16.1)
    Requirement already satisfied: jinja2 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai[tfa]) (3.0.3)
    Requirement already satisfied: jsonschema in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai[tfa]) (4.4.0)
    Requirement already satisfied: entrypoints in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai[tfa]) (0.4)
    Requirement already satisfied: toolz in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai[tfa]) (0.11.2)
    Requirement already satisfied: cffi>=1.12 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from cryptography>=36.0.1->brevettiai[tfa]) (1.15.0)
    Requirement already satisfied: certifi in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from minio<7.1,>=7.0->brevettiai[tfa]) (2021.10.8)
    Requirement already satisfied: urllib3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from minio<7.1,>=7.0->brevettiai[tfa]) (1.26.9)
    Requirement already satisfied: flatbuffers in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from onnxruntime>=1.6.0->brevettiai[tfa]) (1.12)
    Requirement already satisfied: protobuf in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from onnxruntime>=1.6.0->brevettiai[tfa]) (3.19.4)
    Requirement already satisfied: pytz>=2017.2 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from pandas<1.2,>=1.1->brevettiai[tfa]) (2022.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from pandas<1.2,>=1.1->brevettiai[tfa]) (2.8.2)
    Requirement already satisfied: six in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from plotly>=4.14.3->brevettiai[tfa]) (1.16.0)
    Requirement already satisfied: tenacity>=6.2.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from plotly>=4.14.3->brevettiai[tfa]) (8.0.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from pydantic>=1.9.0->brevettiai[tfa]) (4.1.1)
    Requirement already satisfied: idna<4,>=2.5 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from requests>=2.23->brevettiai[tfa]) (3.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from requests>=2.23->brevettiai[tfa]) (2.0.12)
    Requirement already satisfied: scipy>=1.1.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from scikit-learn>=0.23.1->brevettiai[tfa]) (1.7.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from scikit-learn>=0.23.1->brevettiai[tfa]) (3.1.0)
    Requirement already satisfied: joblib>=0.11 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from scikit-learn>=0.23.1->brevettiai[tfa]) (1.1.0)
    Requirement already satisfied: typeguard>=2.7 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from tensorflow_addons>=0.16.1->brevettiai[tfa]) (2.13.3)
    Requirement already satisfied: onnx>=1.4.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from tf2onnx>=1.9.0->brevettiai[tfa]) (1.11.0)
    Requirement already satisfied: pycparser in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from cffi>=1.12->cryptography>=36.0.1->brevettiai[tfa]) (2.21)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jinja2->altair==4.1.0->brevettiai[tfa]) (2.1.1)
    Requirement already satisfied: importlib-metadata in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai[tfa]) (4.11.3)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai[tfa]) (0.18.1)
    Requirement already satisfied: importlib-resources>=1.4.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai[tfa]) (5.4.0)
    Requirement already satisfied: attrs>=17.4.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai[tfa]) (21.4.0)
    Requirement already satisfied: zipp>=3.1.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from importlib-resources>=1.4.0->jsonschema->altair==4.1.0->brevettiai[tfa]) (3.7.0)
    Note: you may need to restart the kernel to use updated packages.



```python
# Platform features useful for accessing the job parameters
from brevettiai.platform import Job, JobSettings
from brevettiai.data.sample_tools import BrevettiDatasetSamples
from brevettiai.interfaces.remote_monitor import RemoteMonitor
# packages model for upload
from brevettiai.utils.model_version import package_saved_model

# Data science tools for training an image classifier model
from brevettiai.data.image import ImageLoader, ImagePipeline
from brevettiai.data.data_generator import DataGenerator, OneHotEncoder
from brevettiai.interfaces.facets_atlas import build_facets

from typing import Optional
from pydantic import Field

class Settings(JobSettings):
    image: ImageLoader = Field(default_factory=ImageLoader,
                               description="Module to facilitate loading of images")
    epochs: int = 2

class TrainingJob(Job):
    settings: Settings

    def run(self):
        # Get the samples and the classes from the job datasets 
        samples = BrevettiDatasetSamples().get_image_samples(self.datasets, annotations=False)
        classes = samples.folder.unique()

        # Setup up data generator to loop through the samples
        # Load images as RGB images
        self.settings.image.channels = 3
        data_generator = DataGenerator(samples, output_structure=("img", "onehot"),
                                       shuffle=True, repeat=True,
                                       max_epoch_samples=4, batch_size=4).map([
            self.settings.image,
            OneHotEncoder(classes=classes, input_key="folder")
        ])

        # Construct a keras image classifier model and train it using the data generator
        model = build_image_classifier(classes, self.settings.image.output_shape())
        # Fit model for user specified number of epochs - remote monitor shows progress on platform
        model.fit(data_generator.get_dataset(), epochs=self.settings.epochs, steps_per_epoch=len(data_generator),
                  callbacks=[self.get_remote_monitor()])

        # *** Building facets for visiualization of output ***
        test_generator = DataGenerator(samples, output_structure=("img"), shuffle=False, repeat=False)\
            .map(self.settings.image)
        samples[classes] = model.predict(test_generator.get_dataset(),
                                         steps=len(test_generator))
        fds = DataGenerator(samples, shuffle=True, output_structure=("img"))\
            .map(ImagePipeline(target_size=(64,64), antialias=True))

        build_facets(fds, job.artifact_path("facets", dir=True), count=32)

        print(f"Facets visible on {self.host_name}/models/{self.id}")
        return model
    
    
    def export(self, model):
        # Save model and package it along with meta data
        # To append metadata overwrite the get_metadata function on your job.
        metadata = self.get_metadata()
        model.save("saved_model", overwrite=True, include_optimizer=False)
        return package_saved_model("saved_model", model_meta=metadata)



```

The training job can be run in a python script; below is shown what is needed to run a training job, and also the TrainingJob class may be used to create the ```settings_schema.json``` file needed by the Brevetti AI platform


```python
import os
import getpass
import sys, json
# Using sagemaker hyperparameters the TrainingJob instantiates
# with settings and dataset access configured by the platform
# Job info: NB: replace with ID and api key from your job
model_id = os.getenv("job_id") or input("Training job model id (can be read from url https://platform.brevetti.ai/models/{model_id})")
api_key = os.getenv("api_key") or getpass.getpass("Training job Api Key:")

job = TrainingJob.init(job_id=model_id, api_key=api_key)
# The train function optimizes the model and returns a path to the model artifact
model_output = job.start(complete_job=False) # The complete_job=False argument lets the platform model job remain open, after testing
model_output
```

    INFO:brevettiai.platform.models.job:<class '__main__.TrainingJob'> initialized
    INFO:brevettiai.platform.models.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json
    INFO:brevettiai.platform.models.dataset:Getting image samples from dataset 'NeurIPS vials TRAIN' [https://platform.brevetti.ai/data/cb14b6e3-b4b9-45bb-955f-47aa6489a192]
    INFO:brevettiai.platform.models.dataset:Contents: {('missing_cap',): 20, ('good',): 20, ('failed_cap',): 19}


    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.


    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.


    Epoch 1/2
    1/1 [==============================] - 6s 6s/step - loss: 4.9836 - accuracy: 0.5000
    Epoch 2/2
    1/1 [==============================] - 1s 666ms/step - loss: 4.6736 - accuracy: 0.5000


    2022-03-23 21:28:48.152412: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
    2022-03-23 21:28:48.182340: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 156905472 exceeds 10% of free system memory.


    WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/tensorflow/python/util/deprecation.py:616: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead


    WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/tensorflow/python/util/deprecation.py:616: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    100%|██████████| 32/32 [00:00<00:00, 587.34it/s]


    Facets visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71





    <keras.engine.sequential.Sequential at 0x7fc643600090>



## Model export
NB: this may take 30 s which is deemed too slow for our test pipelines - therefore the code is commented out


```python
# Export model
#output_path = job.export(model)

# Complete job - which closes the "connection" to the job on the platform, and registers the model for deployment
# - this may be a test job, so please consider which jobs you close

# job.complete_job(output_path)
```

## Create platform settings file: ```settings_schema.json```


```python
schema = Settings.platform_schema().schema
json.dump(schema, open("settings_schema.json", "w"))

# list the output of "running" the script above 
import glob
print("Listing json files in current directory")
glob.glob('./*.json')
```

    Listing json files in current directory





    ['./settings_schema.json']



## Model Zoo: Host training script on [Brevetti AI](https://platform.brevetti.ai)
The training script needs to be packaged in a way that the platform can create new training jobs, to produce new model artifacts for the platform user. To set up the training script for the model zoo, use the web interface [BrevettiAI=>Ressources=>Create model type](https://platform.brevetti.ai/resources/modeltypes/create). This lets you
* Give the model training script a name
* Upload the settings_config.json file
* Link to the docker image with the training code - see below how to create this

![Create model](https://raw.githubusercontent.com/brevettiai/brevettiai-docs/825b6607ee2de6c0c061f503576842f357377792/src/developers/python-sdk-brevettiai/create_model_type.PNG)
![Model zoo](https://raw.githubusercontent.com/brevettiai/brevettiai-docs/825b6607ee2de6c0c061f503576842f357377792/src/developers/python-sdk-brevettiai/model_zoo.PNG)
Screen shot of UI for [Create model: platform.brevetti.ai/resources/modeltypes/create](https://platform.brevetti.ai/resources/modeltypes/create) and [Model Zoo: https://platform.brevetti.ai/models/zoo](https://platform.brevetti.ai/models/zoo)

## Creating the script
When creating a docker image and script file the following needs to be added to the script, to actually execute the code


```python
import sys
sys.argv.append("--serialize_schema")
```


```python
import sys
def main():
    # Run the script with argument --serialize_schema to get a platform settings schema written
    if "--serialize_schema" in sys.argv:
        schema = Settings.platform_schema().schema
        json.dump(schema, open("settings_schema.json", "w"))
    else:
        # Using sagemaker hyperparameters the TrainingJob instantiates
        # with settings and dataset access configured by the platform
        job = TrainingJob.init()
        # The train function optimizes the model and returns a path to the model artifact
        output_path = job.train()
        # The job uploads the model artifact, and closes 
        job.complete_job(output_path)

if __name__ == "__main__":
    # run main module to either serialize settings or run the training
    main()

```

Using the training script, the settings schema file can then be created by simply calling the script with the ```--serialize_schema``` argument.

## A basic docker file
This Dockerfile sets up an environment with tensorflow and install the required ```brevettiai``` package. Furthermore it sets the entrypoint so it will run using sagemaker called by the Brevetti AI platform.

The build script also produces the ```settings_schema.json``` file that is used to create a configurable model for Brevetti AI platform.
```
FROM tensorflow/tensorflow:2.3.1

WORKDIR /brevettiai

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender-dev

COPY image_classifier.py .

# Install the required ```brevettiai``` package
RUN pip install -U git+https://bitbucket.org/criterionai/core#egg=brevettiai[tf2]

# Serializes the settings_schema so it is available in the docker image
RUN python3 image_classifier.py --serialize_schema

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "image_classifier.py"]
```

## Build the docker image and extract the ```settings_schema.json```
If you don't have docker installed, this step can be done on https://labs.play-with-docker.com/

The python image classifier script and the Dockerfile described in the document, can be found on the documentation github repository [image_classifier.py](https://github.com/brevettiai/brevettiai-docs/blob/master/src/tutorials/image_classifier_101/image_classifier.py) and [Dockerfile](https://github.com/brevettiai/brevettiai-docs/blob/master/src/tutorials/image_classifier_101/Dockerfile)

Download the files and run the following lines
```
docker build -f Dockerfile -t image_classifier.docker.image --stream .
```
```
docker create -ti --name ic_container image_classifier.docker.image
docker cp ic_container:/brevettiai/settings_schema.json .

```


## Upload the docker to a container repository
The last step necessary to deploy a docker image is to upload the image to a place that is available to the cloud training job. This can be a public image on [Docker hub](https://dockerhub.io) or a private image hosted by Brevetti AI

It is recommnded to tag the image - the following snippets may be used to push to Dockerhub
```
docker tag $docker_image_name $dockerhub_id/$docker_image_name:$build_number

docker login
docker push $dockerhub_id/$docker_image_name
```

## Create new "model type"
With the built docker image, and the ´´´settings_schema.json´´´ file the scene is set to create a new model on the Brevetti AI platform
[BrevettiAI=>Ressources=>Create model type](https://platform.brevetti.ai/resources/modeltypes/create).

## Create model training job and Test the docker image anywhere
On the [Brevetti AI platform](https://platform.brevetti.ai) you can choose Model=>create model to configure a new model training job, by selecting training data and setting the parameters that the ```settings_schema.json``` specify. The training job is run in the cloud when *Start training* button is pushed.

The script that is created will runs the model training anywhere, if the *model_id* and *api_key* are provided as command arguments.
```
docker run --rm --privileged image_classifier.docker.image hwclock -s  --model_id $your_model_id --api_key $your_api_key
```
NB: these flags hwclock -s --privileged are provided to make the image use the proper clock on windows docker.

## 



To explore the code by examples, please run the in the notebook that can be found on colab on this link [A Brevetti Ai Image Classifier 101](https://githubtocolab.com/brevettiai/brevettiai-docs/blob/master/src/developers/python-sdk-brevettiai/a_brevetti_AI_Image_classifier_101.ipynb)