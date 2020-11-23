#  BrevettiAI Login

High level access for automation of tasks on the platform, tagging, dataset management, models, etc...

## Platform Login
As on the web page you have 15 minutes of access before needing to log back in.


```python
# Imports and setup
from brevettiai.platform import BrevettiAI
web = BrevettiAI()
```


```python
help(web)
```

    Help on PlatformAPI in module brevettiai.platform.web_api object:
    
    class PlatformAPI(builtins.object)
     |  PlatformAPI(username=None, password=None, host=None)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, username=None, password=None, host=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  create_dataset(self, name, tag_ids=None, application=None)
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
     |  delete_tag(self, id)
     |      Delete a tag by id
     |      :param id:
     |      :return:
     |  
     |  download_url(self, url, dst=None, headers=None)
     |  
     |  get_application(self, id)
     |      Get application by id
     |      :param id:
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
     |  get_dataset(self, id=None, raw=True)
     |      Get dataset, or list of all datasets
     |      :param id: guid of dataset (accessible from url on platform) or None for all dataset
     |      :param raw: get as dict, or attempt parsing to Criterion Dataset
     |      :return:
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
     |  get_tag(self, id=None)
     |      Get tag or list of all tags
     |      :param id: tag guid
     |      :return:
     |  
     |  get_userinfo(self)
     |      Get info on user
     |      :return:
     |  
     |  login(self, username, password)
     |  
     |  start_model_training(self, model, submitCloudJob=False)
     |      Start training flow
     |      :param model: model or model id
     |      :param submitCloudJob: submit training to the cloud
     |      :return: updated model
     |  
     |  stop_model_training(self, model, submitCloudJob=False)
     |      Start training flow
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
     |  host
    

