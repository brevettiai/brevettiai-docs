# Basic Image Classifier model example
This document introduces a simple image classifier model, to show how to build the necessary packages to host a model on the Brevetti AI platform. It includes the following necessary steps
* Model training code that can also produce a user configuration ```settings_schema.json``` file
  - should accept cmd args with **job_id** and **api_key** or find sagemaker hyperparameter file with the same
  - should produce a model artifact
  - these steps are handled by the ```brevettiai``` *Job* object
* A simple Dockerfile to run the code - this is how the script is embedded on the platform

## Basic keras Image classifier model
As a minimal model example. The code below will serve as a simple model for image classification based on the MobileNet architecture. It accepts any number of classes and the image size may be specified.
For training regularization, it includes a Dropout layer.
The model and training code can be found on the documentation github repository [image_classifier.py](https://github.com/brevettiai/brevettiai-docs/blob/master/src/tutorials/image_classifier_101/image_classifier.py)


```python
import tensorflow as tf

def build_image_classifier(classes: list, image_shape: tuple):
    # Model backbone is the MobileNetV2
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=image_shape, include_top=False, weights="imagenet"
    )
    # Features are pooled and the output layer consists of a single dense layer
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(len(classes), activation='softmax', name="---".join(classes))
    ])
    # Model is compiled with ```categorical_crossentropy``` loss and reports accuracy metric
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

Using some default image size and classes a model generated with the above code, and we can verify that it seems to produce a valid keras model.


```pyt

To explore the code by examples, please run the in the notebook that can be found on colab on this link [A Brevetti Ai Image Classifier 101](https://githubtocolab.com/brevettiai/brevettiai-docs/blob/master/src/developers/python-sdk-brevettiai/a_brevetti_AI_Image_classifier_101.ipynb)