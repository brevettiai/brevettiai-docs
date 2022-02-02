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

    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Config args found at: 's3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/info.json'


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json


## Job.Settings
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
    <__main__.MyCustomObject object at 0x7fa3607fe410> {'custom_param': 5}


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

    /tmp/config-id-ae03ff72-b7a2-444d-8fe9-623f61dc4c71-kzronpg0/something_i_want_to_save_temporarily.txt
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

