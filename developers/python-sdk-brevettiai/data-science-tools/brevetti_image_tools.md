# Brevetti AI package installation


```python
pip install -U git+https://bitbucket.org/criterionai/core@CORE-22-add-augmentation-to-image-classi
```

    Collecting git+https://bitbucket.org/criterionai/core@CORE-22-add-augmentation-to-image-classiNote: you may need to restart the kernel to use updated packages.
      Cloning https://bitbucket.org/criterionai/core (to revision CORE-22-add-augmentation-to-image-classi) to c:\users\michae~1\appdata\local\temp\pip-req-build-abo0yofa
    Requirement already satisfied, skipping upgrade: backoff>=1.10 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (1.10.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.18 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (1.18.5)
    Requirement already satisfied, skipping upgrade: pandas>=1.0.2 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (1.1.2)
    Requirement already satisfied, skipping upgrade: configparser in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (5.0.0)
    Requirement already satisfied, skipping upgrade: minio>=5.0.10 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (6.0.0)
    Requirement already satisfied, skipping upgrade: requests>=2.23.0 in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from brevetti-ai==1.0) (2.24.0)
    Requirement already satisfied, skipping upgrade: altair==4.1.0 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (4.1.0)
    Requirement already satisfied, skipping upgrade: tqdm in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (4.48.2)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.22 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (0.23.2)
    Requirement already satisfied, skipping upgrade: plotly>=4.6.0 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (4.10.0)
    Requirement already satisfied, skipping upgrade: tensorflow-addons>=0.11.2 in c:\libs\tf2\lib\site-packages (from brevetti-ai==1.0) (0.11.2)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in c:\libs\tf2\lib\site-packages (from pandas>=1.0.2->brevetti-ai==1.0) (2020.1)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in c:\libs\tf2\lib\site-packages (from pandas>=1.0.2->brevetti-ai==1.0) (2.8.1)
    Requirement already satisfied, skipping upgrade: certifi in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from minio>=5.0.10->brevetti-ai==1.0) (2020.6.20)
    Requirement already satisfied, skipping upgrade: urllib3 in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from minio>=5.0.10->brevetti-ai==1.0) (1.25.10)
    Requirement already satisfied, skipping upgrade: idna<3,>=2.5 in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from requests>=2.23.0->brevetti-ai==1.0) (2.10)
    Requirement already satisfied, skipping upgrade: chardet<4,>=3.0.2 in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from requests>=2.23.0->brevetti-ai==1.0) (3.0.4)
    Requirement already satisfied, skipping upgrade: toolz in c:\libs\tf2\lib\site-packages (from altair==4.1.0->brevetti-ai==1.0) (0.10.0)
    Requirement already satisfied, skipping upgrade: jsonschema in c:\libs\tf2\lib\site-packages (from altair==4.1.0->brevetti-ai==1.0) (3.2.0)
    Requirement already satisfied, skipping upgrade: entrypoints in c:\libs\tf2\lib\site-packages (from altair==4.1.0->brevetti-ai==1.0) (0.3)
    Requirement already satisfied, skipping upgrade: jinja2 in c:\libs\tf2\lib\site-packages (from altair==4.1.0->brevetti-ai==1.0) (2.11.2)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in c:\libs\tf2\lib\site-packages (from scikit-learn>=0.22->brevetti-ai==1.0) (0.16.0)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in c:\libs\tf2\lib\site-packages (from scikit-learn>=0.22->brevetti-ai==1.0) (2.1.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from scikit-learn>=0.22->brevetti-ai==1.0) (1.4.1)
    Requirement already satisfied, skipping upgrade: retrying>=1.3.3 in c:\libs\tf2\lib\site-packages (from plotly>=4.6.0->brevetti-ai==1.0) (1.3.3)
    Requirement already satisfied, skipping upgrade: six in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from plotly>=4.6.0->brevetti-ai==1.0) (1.15.0)
    Requirement already satisfied, skipping upgrade: typeguard>=2.7 in c:\libs\tf2\lib\site-packages (from tensorflow-addons>=0.11.2->brevetti-ai==1.0) (2.9.1)
    Requirement already satisfied, skipping upgrade: pyrsistent>=0.14.0 in c:\libs\tf2\lib\site-packages (from jsonschema->altair==4.1.0->brevetti-ai==1.0) (0.16.0)
    Requirement already satisfied, skipping upgrade: attrs>=17.4.0 in c:\libs\tf2\lib\site-packages (from jsonschema->altair==4.1.0->brevetti-ai==1.0) (19.3.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from jsonschema->altair==4.1.0->brevetti-ai==1.0) (1.7.0)
    Requirement already satisfied, skipping upgrade: setuptools in c:\libs\tf2\lib\site-packages (from jsonschema->altair==4.1.0->brevetti-ai==1.0) (49.3.2)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in c:\libs\tf2\lib\site-packages (from jinja2->altair==4.1.0->brevetti-ai==1.0) (1.1.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in c:\users\michael sass hansen\appdata\local\programs\python\python37\lib\site-packages (from importlib-metadata; python_version < "3.8"->jsonschema->altair==4.1.0->brevetti-ai==1.0) (3.1.0)
    Building wheels for collected packages: brevetti-ai
      Building wheel for brevetti-ai (setup.py): started
      Building wheel for brevetti-ai (setup.py): finished with status 'done'
      Created wheel for brevetti-ai: filename=brevetti_ai-1.0-py3-none-any.whl size=77973 sha256=ddcd28cd8eef78a0840f510d8fceb1aad04b546bd7018eb52b5ef07148774c4d
    
    

    WARNING: You are using pip version 20.2.2; however, version 20.2.4 is available.
    You should consider upgrading via the 'C:\libs\tf2\Scripts\python.exe -m pip install --upgrade pip' command.
    

      Stored in directory: C:\Users\MICHAE~1\AppData\Local\Temp\pip-ephem-wheel-cache-c6kho4g8\wheels\ff\55\b4\2886963ba68be1cc1be8f0598be20c08ecc22a6bef06d0bf3f
    Successfully built brevetti-ai
    Installing collected packages: brevetti-ai
      Attempting uninstall: brevetti-ai
        Found existing installation: brevetti-ai 1.0
        Uninstalling brevetti-ai-1.0:
          Successfully uninstalled brevetti-ai-1.0
    Successfully installed brevetti-ai-1.0
    

# Get images from public dataset
Load publicly available dataset


```python
use_dataset = "brevetti_neurips_images"
if use_dataset == "brevetti_neurips_images":
    dataset_path = "s3://public.data.criterion.ai/data/NeurIPS_2018_reduced"
elif use_dataset == "tensorflow_flowers":
    import tensorflow as tf
    dataset_path = str(tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True))
elif use_dataset == "tensorflow_datasets_beans":
    import tensorflow_datasets as tfds
    batch_size = 4

    ds = tfds.load("beans", split="test", shuffle_files=False)

    def encode(x):
        x["encoded"] = tf.io.encode_png(x["image"])
        return x
    def map2float(x):
        x["image"] = tf.cast(x["image"], tf.float32)
        return x
    img_ds = ds.map(encode).map(map2float)
    imgs = next(iter(img_ds.batch(batch_size).take(1)))
    files = []
    for ii in range(batch_size):
        files.append({"path": f"image_{ii}.png"})
        print(f'Writing file {files[-1]["path"]}')
        tf.io.write_file(files[-1]["path"], img["encoded"][ii])

    import pandas as pd
    files = pd.DataFrame(files)

```

## Create Brevetti AI Dataset object to manage listing of data (and access, if it where not publicly available)



```python
from brevettiai.platform.dataset import Dataset
ds = Dataset(bucket=dataset_path, resolve_access_rights=False)

# Fix to get access to a public bucket without credentials
ds.io.minio.client_factory("s3://public.data.criterion.ai", lambda **x:{"endpoint": "s3-eu-west-1.amazonaws.com"})

samples = ds.get_image_samples()
# Printing content of a sample from the pandas data frame
print("Sample: ", samples.sample(1).iloc[0].to_dict())
```

    Sample:  {'category': ('good',), 'folder': 'good', 'path': 's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/11_1543412102711.bmp', 'etag': 'af62217c68ea1dd97a3423d914eeba13', 'bucket': 's3://public.data.criterion.ai/data/NeurIPS_2018_reduced', 'dataset': '', 'dataset_id': '', 'url': 'https://platform.brevetti.ai/download?path=lic.data.criterion.ai%2Fdata%2FNeurIPS_2018_reduced%2Fgood%2F11_1543412102711.bmp'}
    

samples now holds the image samples in a pandas dataframe object. We can investigate the distribution of the different classes


```python
samples.groupby("folder").count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>path</th>
      <th>etag</th>
      <th>bucket</th>
      <th>dataset</th>
      <th>dataset_id</th>
      <th>url</th>
    </tr>
    <tr>
      <th>folder</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bad_cap</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>good</th>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
    </tr>
    <tr>
      <th>missing_cap</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset split - sample integrity module
Functionality to split samples between training and development sets (often referred to as validation set, but this name is confusing in a regulated environment)

This module allows for more fine grained control of the splitting process than what is provided by e.g. sklearn.
The main feature is that it can split based on unique samples rather than just randomly. This is important when multiple images of the same physical item are available


```python
from brevettiai.data.sample_integrity import SampleSplit
from IPython.display import display 

uniqueness_regex = r"/(\d*)_\d*.bmp"

samples = SampleSplit(stratification=["folder"], uniqueness=uniqueness_regex, split=0.8).assign(samples, remainder="devel")
print("Devel samples")
display(samples[samples["purpose"] == "devel"][:5].path.values)
print("Train samples")
display(samples[samples["purpose"] == "train"][:5].path.values)

samples.groupby(["folder", "purpose"]).count()
```

    Devel samples
    


    array(['s3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/1_1543413257224.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/1_1543413266824.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/9_1543413258829.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/9_1543413268396.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/16_1543412094105.bmp'],
          dtype=object)


    Train samples
    


    array(['s3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/0_1543413169486.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/0_1543413189854.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/10_1543413182213.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/10_1543413191789.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/2_1543413180595.bmp'],
          dtype=object)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>category</th>
      <th>path</th>
      <th>etag</th>
      <th>bucket</th>
      <th>dataset</th>
      <th>dataset_id</th>
      <th>url</th>
    </tr>
    <tr>
      <th>folder</th>
      <th>purpose</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bad_cap</th>
      <th>devel</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>train</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">good</th>
      <th>devel</th>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>train</th>
      <td>116</td>
      <td>116</td>
      <td>116</td>
      <td>116</td>
      <td>116</td>
      <td>116</td>
      <td>116</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">missing_cap</th>
      <th>devel</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>train</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



# Data generator
The brevettiai DataGenerator object is a generator object that extents the functionality of tensorflow datasets by adding
* a generated random seed to the map function, so that an image augmentation pipeline may produce reproducible results
* the possibility for stratified sampling such that samples can be drawn with controlled freqeuncy from different groups of the dataset

the method get_dataset() returns a tensorflow dataset object with the above mentioned properties


```python
from brevettiai.data.data_generator import DataGenerator
from brevettiai.data import data_generator

batch_size = 4
# creating a data generator with stratification across a grouping on "folder" and with a weight determined by the square root of number of samples
generator = DataGenerator(samples, batch_size=batch_size, shuffle=True, repeat=True, sampling_groupby=["folder"], sampling_group_weighing="square root")

for sample in generator.get_dataset().take(2):
    print(sample["path"])
```

    tf.Tensor(
    [b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/33_1543412164698.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/28_1543412163686.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/7_1543413267987.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/27_1543412096276.bmp'], shape=(4,), dtype=string)
    tf.Tensor(
    [b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/1_1543413266824.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/30_1543412164091.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/missing_cap/2_1543412677681.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/25_1543412095870.bmp'], shape=(4,), dtype=string)
    

The data generator uses stratified sampling across a grouping on "folder" and with a weight determined by the square root of number of samples.
We can investigate the frequency of samples vs the frequency of actual samples in the dataset


```python
import pandas as pd
from itertools import islice
drawn_samples = pd.DataFrame(islice(generator.get_dataset_numpy(batch=False), len(samples)))
print("Data generator sample frequency")
drawn_samples.groupby("folder").count()
```

    Data generator sample frequency
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>path</th>
      <th>etag</th>
      <th>bucket</th>
      <th>dataset</th>
      <th>dataset_id</th>
      <th>url</th>
      <th>purpose</th>
      <th>_sampling_group</th>
    </tr>
    <tr>
      <th>folder</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bad_cap</th>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
    </tr>
    <tr>
      <th>good</th>
      <td>104</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
    </tr>
    <tr>
      <th>missing_cap</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



# Image pipeline
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

    WARNING:tensorflow:From C:\libs\tf2\lib\site-packages\tensorflow\python\util\deprecation.py:574: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    

# Image augmentation
* transformation augmentation (e.g. flip / rotate / sheare)
* image noise augmentation

Uses a seed so output is repeatable


```python
from brevettiai.data.image.image_augmenter import ImageAugmenter
img_aug = ImageAugmenter()
img_generator_aug = img_generator.map(img_aug)
imgs_aug = next(iter(img_generator_aug))
# The img_generator_aug produces repeatable samples, so taking the first batch a second time, should produce identical output
imgs_aug_repeated = next(iter(img_generator_aug))
```

## Drawing the same sample twice produces the same augmented images


```python
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(3, batch_size, figsize=(16, 12))
for ii in range(batch_size):
    ax[0, ii].imshow(tf.cast(imgs_gen["img"][ii], tf.uint8))
    ax[0, ii].set_title(f"Input image {ii}")
    ax[1, ii].imshow(tf.cast(imgs_aug["img"][ii], tf.uint8))
    ax[1, ii].set_title(f"Augmented image {ii}")
    ax[2, ii].imshow(tf.cast(imgs_aug_repeated["img"][ii], tf.uint8))
    ax[2, ii].set_title(f"Augmented image {ii} repeated")

```


![png](brevetti_image_tools_files/brevetti_image_tools_19_0.png)

