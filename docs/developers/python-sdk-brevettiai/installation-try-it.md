---
Description: How to install the criterion core package
---

# Installation - Try it

## Via git

Installing criterion core via git is done with pip

```bash
pip install brevettiai
```

{% hint style="info" %}
Running in a Jupyter Notebook? add a '!' in front of the commando to run a shell command
{% endhint %}

## Try it in Google Colab hosted Jupyter Notebooks 

We have made a couple of simple Notebooks to run your experiments on the platform. They can be found in the tutorials section.

[Jupyter notebook tutorials](../../../generated/developers/tutorials/tutorials.md)

## Build your own

We have made a small application sample that you can use as a starting point when you are developing models and reporting types to the platform, including an example CI pipeline for bitbucket.

Test them on the platform by looking for the 'Telescope demo classification' and 'Telescope demo segmentation' model types on the platform
[Create model](https://platform.brevetti.ai/models/zoo)

Or get the code yourself and run locally to change it.
[Telescope application bitbucket repository](https://bitbucket.org/criterionai/telescope/src/master/)

```python
from brevettiai.platform import PlatformAPI, PlatformBackend
from telescope import TelescopeSegmentationJob, TelescopeSegmentationSettings
from brevettiai.data.image import CropResizeProcessor
# login with web api
web = PlatformAPI()

# Find dataset to use
datasets = [next(d for d in web.get_dataset() if d.name == "NeurIPS demo vials train")]

# Build settings object
settings = TelescopeSegmentationSettings(
    epochs=50,
    model="Lightning",
    #image=CropResizeProcessor(output_height=224, output_width=224),
    classes=['Cap', 'Thread', 'Neck', 'Container'],
    enable_augmentation=False,
)

# Create model on platform
model = web.create_model("NeurIPS segmentation demo", datasets=datasets, settings=settings)
print(f"{web.host}/models/{model.id}")

# Obtain the job object from the web api (provision your machine)
job = web.initialize_training(model, job_type=TelescopeSegmentationJob)

# Start running the job
job.start()
```

