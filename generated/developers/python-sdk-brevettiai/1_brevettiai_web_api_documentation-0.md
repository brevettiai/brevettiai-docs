#  BrevettiAI Login

High level access for automation of tasks on the platform, tagging, dataset management, models, etc...

## Platform Login
As on the web page you have 60 minutes of access before needing to log back in.


```python
# Imports and setup
from brevettiai.platform import BrevettiAI
import os

model_id = os.getenv("job_id") or input("Training job model id (can be read from url https://platform.brevetti.ai/models/{model_id})")
api_key = os.getenv("api_key")
web = BrevettiAI()

help(web)
```

    2022-01-28 14:19:25.911963: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-01-28 14:19:25.911998: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
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
    

