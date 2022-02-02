# Introduction
The python web api implementation found in the brevettai package gives a lightweight programmatical interface to all the functionality that the [Brevetti AI platform](https://platform.brevetti.ai) offers.

This enables high level access for
* automation of tasks on the platform
* tagging of datasets or models
* dataset management
* managing models etc...

This document shows how this api can be used to get access to datasets and create a model training job.

Web access is granted with your website user, allowing you to automate tasks on the platform. In Python this is achieved through the **BrevettiAI** objec# BrevettiAI Login

High level access for automation of tasks on the platform, tagging, dataset management, models, etc...

## Platform Login
As on the web page you have 60 minutes of access before needing to log back in.


```python
# Imports and setup
from brevettiai.platform import BrevettiAI

web = BrevettiAI()

help(web)
```

    2022-02-02 16:36:01.614703: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-02-02 16:36:01.614731: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7
    /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/tensorflow_addons/utils/ensure_tf_install.py:67: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.5.0 and strictly below 2.8.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.8.0 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      UserWarning,


    Help on PlatformAPI in module brevettiai.platform.web_api object:
    
    class PlatformAPI(builtins.object)
     |  PlatformAPI(username=None, password=None, host=None, remember_me=False)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, username=None, password=None, host=None, remember_me=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  create_dataset(self, name, reference='', notes='', tag_ids=None, application=None)
     |      Create dataset on platform
     |      :param name: Name of dataset
     |      :param tag_ids:
     |      :param application:
     |      :return:
     |  
     |  create_model(self, name, model_type, settings, datasets, tags=[], application=None)
     |      Create new model
     |      :param name:
     |      :param model_type:
     |      :param settings:
     |      :param datasets:
     |      :param tags:
     |      :param application:
     |      :param schema:
     |      :return:
     |  
     |  create_tag(self, name, parent_id=None)
     |      Create a tag on the platform
     |      :param name:
     |      :param parent_id: guid of tag parent or none to create root tag
     |      :return:
     |  
     |  create_testreport(self, model_id, datasets)
     |      Create new test report
     |      :param model_id:
     |      :param datasets:
     |      :return:
     |  
     |  delete_model(self, id=None)
     |      Get model or list of all models
     |      :param id: Guid of model (available in the url), or None
     |      :return:
     |  
     |  delete_sftp_user(self, dataset, user)
     |  
     |  delete_tag(self, id)
     |      Delete a tag by id
     |      :param id:
     |      :return:
     |  
     |  download_url(self, url, dst=None, headers=None)
     |  
     |  get_application(self, id=None, model_id=None)
     |      Get application by id
     |      :param id: either application id or model id
     |      :return:
     |  
     |  get_artifacts(self, model_id, prefix='', type='models')
     |      Get artifacts for model or test report
     |      :param model_id: Guid of model/test report (available in the url)
     |      :param prefix:
     |      :param type: 'models' / 'reports'
     |      :return:
     |  
     |  get_available_model_types(self)
     |      List all available model types
     |      :return:
     |  
     |  get_dataset(self, id=None, raw=False, write_access=False)
     |      Get dataset, or list of all datasets
     |      :param id: guid of dataset (accessible from url on platform) or None for all dataset
     |      :param raw: get as dict, or attempt parsing to Criterion Dataset
     |      :return:
     |  
     |  get_dataset_sts_assume_role_response(self, guid)
     |  
     |  get_devices(self, id=None)
     |  
     |  get_endpoint(self, endpoint, **kwargs)
     |  
     |  get_model(self, id=None)
     |      Get model or list of all models
     |      :param id: Guid of model (available in the url), or None
     |      :return:
     |  
     |  get_modeltype(self, id=None)
     |      Grt type of model
     |      :param id: model guid
     |      :return:
     |  
     |  get_project(self, id=None)
     |  
     |  get_report(self, id=None)
     |      Get test report, or list of all reports
     |      :param id: Guid of test report (available in the url), or None
     |      :return:
     |  
     |  get_schema(self, modeltype)
     |      Get schema for a certain model type
     |      :param modeltype: model type or guid
     |      :return:
     |  
     |  get_sftp_user(self, dataset, **kwargs)
     |  
     |  get_tag(self, id=None)
     |      Get tag or list of all tags
     |      :param id: tag guid
     |      :return:
     |  
     |  get_userinfo(self)
     |      Get info on user
     |      :return:
     |  
     |  login(self, username, password, remember_me=False)
     |  
     |  start_model_training(self, model, submitCloudJob=False)
     |      Start training flow
     |      :param model: model or model id
     |      :param submitCloudJob: submit training to the cloud
     |      :return: updated model
     |  
     |  stop_model_training(self, model, submitCloudJob=False)
     |      Stop training flow
     |      :param model: model or model id
     |      :param submitCloudJob: submit training to google cloud
     |      :return: updated model
     |  
     |  update_dataset(self, id, *, name: str, reference: str, notes: str, tags: list, locked: bool)
     |      Update dataset on platform
     |      :param id: guid of dataset
     |      :param name: dataset name
     |      :param reference: dataset reference
     |      :param notes: dataset notes
     |      :param tags: list of tag ids (each represented by a string)
     |      :param locked: sets the lock status of the dataset
     |      :return:
     |  
     |  update_dataset_permission(self, id, userId, groupId=None, permissionType='Editor')
     |      Update dataset permissions for user
     |      :param id:
     |      :param userId:
     |      :param groupId:
     |      :param permissionType:
     |      :return:
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  antiforgery_headers
     |      Get anti forgery headers from platform
     |      :return:
     |  
     |  backend
     |  
     |  host
    



```python
?web
```

# Element acces, list datasets, tags, models...

with the web object you can list, manage and edit elements on the web page.
Most of the functions require an id, the guid of the object to manipulate. Alternatively to get all use None as id.

EX: to list datasets, tags, and models, call get_... with no id (id=None)


```python
datasets = web.get_dataset()
tags = web.get_tag()
models = web.get_model()

# List 10 first dataset names
[d.name for d in datasets][:10]
```




    ['NeurIPS 2018',
     'NeurIPS 2018 large',
     'Blood Cell Images',
     'Agar plates',
     'NeurIPS vials TRAIN']



For a single dataset, model or ... use the get_... functions with an id


```python
dataset = web.get_dataset(datasets[0].id)
dataset
```




    <brevettiai.platform.dataset.Dataset at 0x7fd5b2bb4a90>



# Customized Job Settings
Settings are esstentially the serialized configuration of a training job algorithm.
Settings can be used for a training job configuration by letting a user change settings, and settings are included in the default job output, such that the parameters of a training job can be saved and tracked for comparison and audit purposes.


```python
from brevettiai.interfaces.vue_schema_utils import VueSettingsModule as SettingsModule

class MyAlgoObject(SettingsModule):
    def __init__(self, multiply_factor: float = 2.0, 
                 enable : bool = True):
        self.multiply_factor = multiply_factor
        self.enable = enable
    def __call__(self, x):
        factor = 1.0
        if self.enable:
            factor *= self.multiply_factor
        return x * factor
test_obj = MyAlgoObject(multiply_factor=3.0)

# Settings used for creating the job
settings = test_obj.get_settings()
print(settings)
```

    {'multiply_factor': 3.0, 'enable': True}


# Create Model Training Job
To enter the job context you can either create a model on the platform or programatically via the web api.

The following code finds the firs dataset and creates a model (job) with access to this model.
The model context type is the id of a model type on the platform to use.
After running the model is available on the website, along with an s3 bucket for artifacts for your job outputs


When creating a model you have the option to include datasets and tags and settings defining your model.


```python
# Datasets to add to the created job
datasets = web.get_dataset()[1:2]

model_context_type = "a0aaad69-c032-41c1-a68c-e9a15a5fb18c" # "Magic" undocumented uId of *external* job model type

model_def = web.create_model(name=f'Test {web.user["firstName"]} {web.user["lastName"]}',
                             model_type=model_context_type,
                             settings=settings,
                             datasets=datasets)
```

## Start job

The model id and the model api key gives you access to use the python sdk to access data, and to upload artifacts and lots of other cool stuff. To enable this, we need to start model training - this is the same as selecting "Simulate training" on the platform.


```python
# Starting training in simulate mode
web.start_model_training(model=model_def['id'])
print(f"Model url: {web.host}/models/{model_def['id']} (Please check it out :)\n")
print("To access data and model through python SDK use the following")
print(f"Model id: {model_def['id']}")
print(f"Model api key: {model_def['apiKey']}")
```

    Model url: https://platform.brevetti.ai/models/ae32ec95-f462-452f-b0aa-37a7bd59ea8e (Please check it out :)
    
    To access data and model through python SDK use the following
    Model id: ae32ec95-f462-452f-b0aa-37a7bd59ea8e
    Model api key: AeMVz0d03h7KQ3JkmUj2yOAM


## NB: Delete job
If the job has not been deployed, and you are e.g. just testing interfaces, you may delete a job


```python
# NB: delete model, there is no simple "undo" funcionality for this
web.delete_model(id=model_def['id'])
```




    <Response [204]>




To explore the code by examples, please run the in the notebook that can be found on colab on this link [1 Brevettiai Web Api Documentation](https://githubtocolab.com/brevettiai/brevettiai-docs/blob/master/src/developers/python-sdk-brevettiai/1_brevettiai_web_api_documentation.ipynb)