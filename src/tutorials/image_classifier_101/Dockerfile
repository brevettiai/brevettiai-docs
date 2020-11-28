FROM tensorflow/tensorflow:2.3.1

WORKDIR /brevettiai

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender-dev

COPY image_classifier.py .

# Install the required ```brevettiai``` package
RUN pip install -U git+https://bitbucket.org/criterionai/core#egg=brevettiai[tf2]

# Serializes the settings_schema so it is available in the docker image
RUN python3 image_classifier.py --serialize_schema

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "image_classifier.py"]
