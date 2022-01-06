---
Description: 'This section describes how to create, train, and test models in the Platform.'
---

# Introduction
The model-training-and-testing section describes how to set up a model, train it and finally test it using the Brevetti AI platform.
This tutorial will be based on the use of Brevetti AI's internal repository, image-segmentation, aswell as the open-source library Criterion-core and tensorflow-onnx.

## Set up in a virtual environment
To run a model training locally, the following open-source repositories must first be cloned locally, preferably to a virtual environment, and subsequently pip installed. Remember to use the branch "development" if you want to test locally.

* https://bitbucket.org/criterionai/core/src/development/
* https://github.com/brevettiai/tensorflow-onnx

If using the image-segmentation repository
* https://bitbucket.org/criterionai/image-segmentation/src/development/

When the repositories are cloned to your virtual environment, they should follow a folder structure like this:
- python/criterionai/image_segmentation
- python/criterionai/core
- python/criterionai/tensorflow-onnx
- ...

Now the repositories are pip installed, and it is important to do it in the following order, or it may not work:
(remember to use the -e flag if possible when pip installing the repo's).
1. pip install nose coverage
2. pip install -e ./core
3. pip install -e ./tensorflow-onnx
4. pip install -e ./image-segmentation

If using the image-segmentation repo, it may fail when trying to import brevettiai package, if your associated account do not have access. This will also cause it to fail installing some libraries correctly, which it would otherwise do automatically, such as seaborn and shapely. These will have to be installed manually in that case. The brevettiai package is already installed through core, so that is alright.

There might also be pip dependency conflicts between the packages, such as the angus package requiring a more modern version of scikit-learn for example, while brevettiai-1.0 requiring an earlier version. If this causes problems later on, try playing around with upgrading or backrolling the packages whose current version gives cause of error.

## Create a model
To set up a model in the platform...

## Model training
Once a model has been set up, it can be 

## Model testing
