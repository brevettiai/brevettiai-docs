# Introduction
The python web api implementation found in the brevettai package gives a lightweight programmatical interface to all the functionality that the [Brevetti AI platform](https://platform.brevetti.ai) offers.

This enables high level access for
* automation of tasks on the platform
* tagging of datasets or models
* dataset management
* managing models etc...

This document shows how this api can be used to get access to datasets and create a model training job.

Web access is granted with your website user, allowing you to automate tasks on the platform. In Python this is achieved through the **BrevettiAI** object
# BrevettiAI Login

High level access for automation of tasks on the platform, tagging, dataset management, models, etc...

## Platform Login
As on the web page you have 60 minutes of access before needing to log back in.


```python
# Imports and setup
from brevettiai.platform import BrevettiAI

web = BrevettiAI()

help(web)
```

    2022-03-03 14:09:23.335655: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-03-03 14:09:23.335688: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7


    Help on PlatformAPI in module brevettiai.platform.web_api object:
    
    class PlatformAPI(builtins.object)
     |  PlatformAPI(username=None, password=None, host=None, remember_me=False)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, username=None, password=None, host=None, remember_me=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  create(self, obj: Union[brevettiai.platform.models.dataset.Dataset, brevettiai.platform.models.tag.Tag, brevettiai.platform.models.web_api_types.Model, brevettiai.platform.models.web_api_types.Report], **kwargs)
     |  
     |  create_model(self, name, datasets, settings: brevettiai.platform.models.job.JobSettings = None, model_type=None, tags=None, application: brevettiai.platform.models.web_api_types.Application = None)
     |      Create new model
     |      :param name:
     |      :param model_type:
     |      :param settings:
     |      :param datasets:
     |      :param tags:
     |      :param application:
     |      :return:
     |  
     |  create_sftp_user(self, dataset, **kwargs) -> brevettiai.platform.models.web_api_types.SftpUser
     |  
     |  create_testreport(self, name, model, datasets, report_type, settings, tags, submitToCloud=False)
     |      Create new test report
     |      :param name:
     |      :param model:
     |      :param datasets:
     |      :param report_type:
     |      :param settings:
     |      :param tags:
     |      :param submitToCloud:
     |      :return:
     |  
     |  delete(self, obj: Union[brevettiai.platform.models.dataset.Dataset, brevettiai.platform.models.tag.Tag, brevettiai.platform.models.web_api_types.Model, brevettiai.platform.models.web_api_types.Report, brevettiai.platform.models.web_api_types.SftpUser])
     |  
     |  download_url(self, url, dst=None, headers=None)
     |  
     |  get_application(self, id=None) -> Union[brevettiai.platform.models.web_api_types.Application, List[brevettiai.platform.models.web_api_types.Application]]
     |      Get application by id
     |      :param id: either application id or associated id (model, dataset)
     |      :return:
     |  
     |  get_artifacts(self, obj: Union[brevettiai.platform.models.web_api_types.Model, brevettiai.platform.models.web_api_types.Report], prefix: str = '')
     |      Get artifacts for model or test report
     |      :param obj: model/test report object
     |      :param prefix: object prefix (folder)
     |      :return:
     |  
     |  get_available_model_types(self)
     |      List all available model types
     |      :return:
     |  
     |  get_dataset(self, id=None, write_access=False) -> Union[brevettiai.platform.models.dataset.Dataset, List[brevettiai.platform.models.dataset.Dataset]]
     |      Get dataset, or list of all datasets
     |      :param id: guid of dataset (accessible from url on platform) or None for all dataset
     |      :param write_access: Assume read and write access to dataset
     |      :return:
     |  
     |  get_dataset_sts_assume_role_response(self, guid)
     |  
     |  get_device(self, id=None)
     |  
     |  get_model(self, id=None) -> Union[brevettiai.platform.models.web_api_types.Model, List[brevettiai.platform.models.web_api_types.Model]]
     |      Get model or list of all models
     |      :param id: Guid of model (available in the url), or None
     |      :return:
     |  
     |  get_modeltype(self, id=None, master=False) -> Union[brevettiai.platform.models.web_api_types.ModelType, List[brevettiai.platform.models.web_api_types.ModelType]]
     |      Grt type of model
     |      :param id: model guid
     |      :param master: get from master
     |      :return:
     |  
     |  get_project(self, id=None) -> Union[brevettiai.platform.models.web_api_types.Project, List[brevettiai.platform.models.web_api_types.Project]]
     |  
     |  get_report(self, id=None) -> Union[brevettiai.platform.models.web_api_types.Report, List[brevettiai.platform.models.web_api_types.Report]]
     |      Get test report, or list of all reports
     |      :param id: Guid of test report (available in the url), or None
     |      :return:
     |  
     |  get_reporttype(self, id=None, master=False) -> Union[brevettiai.platform.models.web_api_types.ReportType, List[brevettiai.platform.models.web_api_types.ReportType]]
     |      Grt type of model
     |      :param id: model guid
     |      :param master: get from master
     |      :return:
     |  
     |  get_schema(self, obj: Union[brevettiai.platform.models.web_api_types.ModelType, brevettiai.platform.models.web_api_types.ReportType])
     |      Get schema for a certain model type
     |      :param obj modeltype or report type
     |      :return:
     |  
     |  get_sftp_users(self, dataset, **kwargs) -> List[brevettiai.platform.models.web_api_types.SftpUser]
     |  
     |  get_tag(self, id=None) -> Union[brevettiai.platform.models.tag.Tag, List[brevettiai.platform.models.tag.Tag]]
     |      Get tag or list of all tags
     |      :param id: tag guid
     |      :return:
     |  
     |  get_userinfo(self)
     |      Get info on user
     |      :return:
     |  
     |  initialize_training(self, model: Union[str, brevettiai.platform.models.web_api_types.Model], job_type: Type[brevettiai.platform.models.job.Job] = None, submitToCloud=False) -> Union[brevettiai.platform.models.job.Job, NoneType]
     |      Start training flow
     |      :param model: model or model id
     |      :param job_type:
     |      :param submitToCloud: submit training to the cloud
     |      :return: updated model
     |  
     |  login(self, username, password, remember_me=False)
     |  
     |  stop_model_training(self, model)
     |      Stop training flow
     |      :param model: model
     |      :return: updated model
     |  
     |  update(self, obj, master=False)
     |  
     |  update_dataset_permission(self, id, user_id, group_id=None, permission_type='Editor')
     |      Update dataset permissions for user
     |      :param id:
     |      :param user_id:
     |      :param group_id:
     |      :param permission_type:
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
     |  
     |  io
    



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




    Dataset(id='21263db3-1c9b-456b-be1d-ecfa2afb5d99', bucket='s3://data.criterion.ai/21263db3-1c9b-456b-be1d-ecfa2afb5d99', name='NeurIPS 2018', locked=False, reference='Batch HW0001', notes='', tags=[])



# Customized Job Settings
Settings are esstentially the serialized configuration of a training job algorithm.
Settings can be used for a training job configuration by letting a user change settings, and settings are included in the default job output, such that the parameters of a training job can be saved and tracked for comparison and audit purposes.


```python
from brevettiai import Job, JobSettings
        
class MyAlgoObject(JobSettings):
    multiply_factor: float = 2.0
    enable : bool = True

    def __call__(self, x):
        factor = 1.0
        if self.enable:
            factor *= self.multiply_factor
        return x * factor
test_obj = MyAlgoObject(multiply_factor=3.0)

# Settings used for creating the job
print(test_obj, test_obj(2))
```

    extra={} multiply_factor=3.0 enable=True 6.0



```python
class MyJob(Job):
    settings: MyAlgoObject
    
    def run(self): # This function should be overloaded and is run when job is started

        print(f"Run my custom code using custom parameters : {self.settings.__dict__}")
        print(f"Result on input 2.0: {self.settings(2.0)}")
        return None # Return path to model artifacts to be uploaded after job is completed
        

```

# Create Model Training Job
To enter the job context you can either create a model on the platform or programatically via the web api.

The following code finds the firs dataset and creates a model (job) with access to this model.
The model context type is the id of a model type on the platform to use.
After running the model is available on the website, along with an s3 bucket for artifacts for your job outputs


When creating a model you have the option to include datasets and tags and settings defining your model.


```python
# Datasets to add to the created job
datasets = web.get_dataset()[1:2]

model = web.create_model(name=f'Test {web.user["firstName"]} {web.user["lastName"]}',
                         settings=test_obj,
                         datasets=datasets)
```

## Start job

The model id and the model api key gives you access to use the python sdk to access data, and to upload artifacts and lots of other cool stuff. To enable this, we need to start model training - this is the same as selecting "Simulate training" on the platform.


```python
# Starting training in simulate mode
job = web.initialize_training(model=model, job_type=MyJob)
print(f"Model url: {web.host}/models/{model.id} (Please check it out :)\n")
print("To access data and model through python SDK use the following")
print(f"Model id: {model.id}")
print(f"Model api key (invalid when job is completed, or model i deleted)): {model.api_key}")
```

    INFO:brevettiai.platform.models.job:<class '__main__.MyJob'> initialized


    Model url: https://platform.brevetti.ai/models/094d406d-96c4-4c7b-8cd3-1ddb139d988d (Please check it out :)
    
    To access data and model through python SDK use the following
    Model id: 094d406d-96c4-4c7b-8cd3-1ddb139d988d
    Model api key (invalid when job is completed, or model i deleted)): 0YgQu4PaA4ce37HX3VEg9SwO



```python
job.start()
```

    INFO:brevettiai.platform.models.job:Uploading output.json to s3://data.criterion.ai/094d406d-96c4-4c7b-8cd3-1ddb139d988d/artifacts/output.json
    INFO:brevettiai.platform.models.job:Uploading output.json to s3://data.criterion.ai/094d406d-96c4-4c7b-8cd3-1ddb139d988d/artifacts/output.json


    Run my custom code using custom parameters : {'extra': {'extra': {}}, 'multiply_factor': 3.0, 'enable': True}
    Result on input 2.0: 6.0


    INFO:brevettiai.platform.models.job:Job completed: modelPath=


## NB: Delete job
If the job has not been deployed, and you are e.g. just testing interfaces, you may delete a job


```python
# NB: delete model, there is no simple "undo" funcionality for this
web.delete(model)
```



To explore the code by examples, please run the in the notebook that can be found on colab on this link [1 Brevettiai Web Api Documentation](https://githubtocolab.com/brevettiai/brevettiai-docs/blob/master/src/developers/python-sdk-brevettiai/1_brevettiai_web_api_documentation.ipynb)