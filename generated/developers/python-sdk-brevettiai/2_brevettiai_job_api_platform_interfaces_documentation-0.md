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

    DEBUG:tensorflow:Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud-tpu-client.


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Config args found at: 's3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/info.json'


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json


## Storage

In the job context you have two storage modes, temporary and persisted storage. Temporary storage is local on the machine, while the persisted storage is in the cloud in the form of artifacts.


```python
temp_path = job.temp_path("something_i_want_to_save_temporarily.txt")
print(temp_path)

job.io.write_file(temp_path, "Valuable information")
print(str(job.io.read_file(temp_path), "utf-8"))
```

    /tmp/config-id-ae03ff72-b7a2-444d-8fe9-623f61dc4c71-sc64ue46/something_i_want_to_save_temporarily.txt
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

