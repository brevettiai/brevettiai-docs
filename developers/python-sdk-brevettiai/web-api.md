# Web API

The Criterion Core web api is a lightweight python wrapper on top of the platform APIs documented in [Swagger](https://app.criterion.ai/swagger/index.html).

## Log in

To access the API use CriterionWeb to log in

```text
from brevettiai.platform import BrevettiAI
web = BrevettiAI()
```

## Get elements

From then on you can get your datasets, tags an models with one line commands

```text
datasets, tags, models = web.get_dataset(), web.get_tag(), web.get_model()
```

Or single items if you know the GUID, accessible from the platform URL

```text
dataset = web.get_dataset(id="dcb9f94b-25c4-4c7c-b5a1-ba3ccd65f358")
```

## Create Models

Create a model automatically

```text
# Datasets the model should have access to
datasets = web.get_dataset()[:1]
# Settings for the model
settings = dict(some_setting_i_need=1)

# Id of 'external job' model type.
model_context_type = "a0aaad69-c032-41c1-a68c-e9a15a5fb18c"
model_def = web.create_model(f'Test {web.user["firstName"]} {web.user["lastName"]}', model_context_type,
                             settings=settings, datasets=datasets)
print(f"URL: {web.baseurl}/models/{model_def['id']}")
```

With the job can be started from your script

```text
# Starting training in simulate mode
web.start_model_training(model=model_def['id'])

print(f"Model url: {web.baseurl}/models/{model_def['id']} (Please check it out :)\n")
print("To access data and model through python SDK use the following")
print(f"Model id: {model_def['id']}")
print(f"Model api key: {model_def['apiKey']}")
```

