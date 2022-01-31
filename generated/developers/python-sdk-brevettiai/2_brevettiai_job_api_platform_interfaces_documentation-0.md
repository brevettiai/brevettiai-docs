#   BrevettiAI Platform Job Interface
To access a job you need the model id from e.g. the model path https://platform.brevetti.ai/models/<model_id> and the api key, also accessible from the platform, which together grant you access to the data storage ressources.

If you, instead, used the web api to get access to, and start a model, they id and key can be found in the response

* model_id = model_def["id"]

* api_key = model_def["apiKey"]



```python
# Job info: NB: replace with ID and api key from your job
import os
import getpass

model_id = os.getenv("job_id") or input("Training job model id (can be read from url https://platform.brevetti.ai/models/{model_id})")
api_key = os.getenv("api_key") or getpass.getpass("Training job Api Key:")
```


```python
from brevettiai.platform import Job
from brevettiai.interfaces import vue_schema_utils
 
job = Job.init(job_id=model_id, api_key=api_key)
```

    2022-01-28 14:19:40.316001: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-01-28 14:19:40.316039: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    DEBUG:tensorflow:Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud-tpu-client.
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Config args found at: 's3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/info.json'


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json


## Settings
The *brevettiai* library contains an interface to easily serialize and deserialize objects in json format. This has several advantages in designing image computer vision pipelines
* It is easy to alter parameters when running new training jobs / experiments
* The training process parameters can stored for each experiment to keep track of experiments


```python
# Job info: NB: replace with ID and api key from your job
from brevettiai.interfaces.vue_schema_utils import VueSettingsModule


class MyCustomObject(VueSettingsModule):
    def __init__(self, custom_param: int = 0):
        self.custom_param = custom_param


class SerializeableObject(VueSettingsModule):
    def __init__(self, param_1: float = -1.0, param_2: str = None, param_3: dict = None, custom: MyCustomObject = None):
        self.param_1 = param_1
        self.param_2 = param_2 or "Param2 not provided"
        self.param_3 = param_3 or {"nested_param"}
        self.custom = custom or MyCustomObject()

my_object = SerializeableObject(param_1=1, param_2="Test", param_3={"test_dict_serialization": "nested"})
# Serialize objects with get_config
config = my_object.get_config()

print("Original object config: \n", config)

# Deserialize objects with from_config
my_object_copy = SerializeableObject.from_config(config)

print("Deserialized object config: \n", my_object_copy.get_config())
```

    Original object config: 
     {'param_1': 1, 'param_2': 'Test', 'param_3': {'test_dict_serialization': 'nested'}, 'custom': {'custom_param': 0}}
    Deserialized object config: 
     {'param_1': 1, 'param_2': 'Test', 'param_3': {'test_dict_serialization': 'nested'}, 'custom': {'custom_param': 0}}


### Job.Settings
parsing settings to a training job using command line arguments


```python
from brevettiai.platform import Job
import sys

# Parsing parameters using command line args will set the settings in the nested object
# job.settings

# For classes hinted to be an object type as 'dict', 'list' etc the parameter text will be json parsed

sys.argv += ["--param_3", '{"test_dict_serialization": "nested2"}',
             "--custom.custom_param", "5"]

class MiniJobWithCustomSettings(Job):
    Settings = SerializeableObject
    settings: Settings

job = MiniJobWithCustomSettings.init(job_id=model_id, api_key=api_key)
```

    INFO:brevettiai.platform.job:Config args found at: 's3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/info.json'
    WARNING:brevettiai.utils.module:Invalid config keys: some_setting_i_need
    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json



```python
print(job.settings.get_config()) # NB: param_3 and custom has been initialized by the command line parameters assigned above
print(job.settings.custom, job.settings.custom.__dict__)
```

    {'param_1': -1.0, 'param_2': 'Param2 not provided', 'param_3': {'test_dict_serialization': 'nested2'}, 'custom': {'custom_param': 5}}
    <__main__.MyCustomObject object at 0x7f097aa73cd0> {'custom_param': 5}


The following will upload a serialized version of the training pipeline whenever a job is run


```python
job.upload_job_output()
```

    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json





    'artifacts/output.json'



## Storage

In the job context you have two storage modes, temporary and persisted storage. Temporary storage is local on the machine, while the persisted storage is in the cloud in the form of artifacts.


```python
temp_path = job.temp_path("something_i_want_to_save_temporarily.txt")
print(temp_path)

job.io.write_file(temp_path, "Valuable information")
print(str(job.io.read_file(temp_path), "utf-8"))
```

    /tmp/config-id-ae03ff72-b7a2-444d-8fe9-623f61dc4c71-rcbs4udb/something_i_want_to_save_temporarily.txt
    Valuable information



```python
artifact_path = job.artifact_path("something_i_want_to_save.txt")
print(f"Available at on the website: {job.host_name}/models/{job.id}/artifacts")

# And in from the job
job.io.write_file(artifact_path, "Valuable information")
print(str(job.io.read_file(artifact_path), "utf-8"))

```

    Available at on the website: https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts
    Valuable information

