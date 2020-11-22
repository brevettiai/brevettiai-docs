#  Image pipeline
The ImagePipeline object is a utility for
* reading a wide range of image formats and adding the reader to the tensorflow dataset graph
* (optionally) select region(s) of interest
* (optionally) rescale / pad the image to the desired output shape



```python
from brevettiai.data.image.image_pipeline import ImagePipeline

pipeline = ImagePipeline(target_size=(128, 128))
img_generator = generator.map(pipeline)

#The image generator now adds the loaded (and reshaped) image to the dataset execution graph, and per default the output is added using the "img" key

imgs_gen = next(iter(img_generator))
# imgs_gen now holds samples with an added image
```

    WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.7.9/x64/lib/python3.7/site-packages/tensorflow/python/util/deprecation.py:574: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead

