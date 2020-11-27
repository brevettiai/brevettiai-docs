FROM tensorflow/tensorflow:2.3.1

WORKDIR /brevettiai

#apt-get update && 
RUN apt-get install -y git libsm6 libxext6 libxrender-dev

COPY image_classifier.py .

RUN pip install -U git+https://bitbucket.org/criterionai/core#egg=brevettiai[tf2]
RUN python3 image_classifier.py --serialize_schema
RUN echo ls
# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "image_classifier.py"]