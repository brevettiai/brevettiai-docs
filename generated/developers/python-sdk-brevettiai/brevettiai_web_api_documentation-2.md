#  Create Job
To enter the job context you can either create a model on the platform or programatically via the web api.

The following code finds the firs dataset and creates a model (job) with access to this model.
The model context type is the id of a model type on the platform to use.
After running the model is available on the website, along with an s3 bucket for artifacts for your job outputs


When creating a model you have the option to include datasets and tags and settings defining your model.


```python
# Datasets to add to the created job
datasets = web.get_dataset()[1:2]

# Settings used for the job
settings = dict(some_setting_i_need=1)

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

    Model url: https://platform.brevetti.ai/models/c85085bb-34be-4d58-8390-b69055bbb207 (Please check it out :)
    
    To access data and model through python SDK use the following
    Model id: c85085bb-34be-4d58-8390-b69055bbb207
    Model api key: yEYP3D3qaSpo7s87HT5aG00W

