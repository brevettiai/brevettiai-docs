```python
pip install brevettiai
```

    Requirement already satisfied: brevettiai in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (0.2.2)
    Requirement already satisfied: tf2onnx>=1.9.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.9.3)
    Requirement already satisfied: pandas<1.2,>=1.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.1.5)
    Requirement already satisfied: backoff>=1.10 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.11.1)
    Requirement already satisfied: tqdm>=4.62 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (4.62.3)
    Requirement already satisfied: pydantic>=1.8.2 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.9.0)
    Requirement already satisfied: numpy>=1.18.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.21.5)
    Requirement already satisfied: altair==4.1.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (4.1.0)
    Requirement already satisfied: plotly>=4.14.3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (5.5.0)
    Requirement already satisfied: configparser>=5.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (5.2.0)
    Requirement already satisfied: requests>=2.23 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (2.27.1)
    Requirement already satisfied: scikit-learn>=0.23.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.0.2)
    Requirement already satisfied: cryptography<37.0.0,>=36.0.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (36.0.1)
    Requirement already satisfied: minio<7.1,>=7.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (7.0.4)
    Requirement already satisfied: shapely>=1.7.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (1.8.0)
    Requirement already satisfied: mmh3>=3.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from brevettiai) (3.0.0)
    Requirement already satisfied: jinja2 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai) (3.0.3)
    Requirement already satisfied: entrypoints in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai) (0.3)
    Requirement already satisfied: jsonschema in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai) (4.4.0)
    Requirement already satisfied: toolz in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from altair==4.1.0->brevettiai) (0.11.2)
    Requirement already satisfied: cffi>=1.12 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from cryptography<37.0.0,>=36.0.1->brevettiai) (1.15.0)
    Requirement already satisfied: certifi in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from minio<7.1,>=7.0->brevettiai) (2021.10.8)
    Requirement already satisfied: urllib3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from minio<7.1,>=7.0->brevettiai) (1.26.8)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from pandas<1.2,>=1.1->brevettiai) (2.8.2)
    Requirement already satisfied: pytz>=2017.2 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from pandas<1.2,>=1.1->brevettiai) (2021.3)
    Requirement already satisfied: tenacity>=6.2.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from plotly>=4.14.3->brevettiai) (8.0.1)
    Requirement already satisfied: six in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from plotly>=4.14.3->brevettiai) (1.16.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from pydantic>=1.8.2->brevettiai) (4.0.1)
    Requirement already satisfied: idna<4,>=2.5 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from requests>=2.23->brevettiai) (3.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from requests>=2.23->brevettiai) (2.0.11)
    Requirement already satisfied: joblib>=0.11 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from scikit-learn>=0.23.1->brevettiai) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from scikit-learn>=0.23.1->brevettiai) (3.1.0)
    Requirement already satisfied: scipy>=1.1.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from scikit-learn>=0.23.1->brevettiai) (1.7.3)
    Requirement already satisfied: flatbuffers~=1.12 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from tf2onnx>=1.9.0->brevettiai) (1.12)
    Requirement already satisfied: onnx>=1.4.1 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from tf2onnx>=1.9.0->brevettiai) (1.10.2)
    Requirement already satisfied: pycparser in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from cffi>=1.12->cryptography<37.0.0,>=36.0.1->brevettiai) (2.21)
    Requirement already satisfied: protobuf in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from onnx>=1.4.1->tf2onnx>=1.9.0->brevettiai) (3.19.4)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jinja2->altair==4.1.0->brevettiai) (2.0.1)
    Requirement already satisfied: importlib-resources>=1.4.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai) (5.4.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai) (0.18.1)
    Requirement already satisfied: attrs>=17.4.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai) (21.4.0)
    Requirement already satisfied: importlib-metadata in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from jsonschema->altair==4.1.0->brevettiai) (4.10.1)
    Requirement already satisfied: zipp>=3.1.0 in /opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages (from importlib-resources>=1.4.0->jsonschema->altair==4.1.0->brevettiai) (3.7.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import os
import getpass
model_id = os.getenv("job_id") or input("Training job model id (can be read from url https://platform.brevetti.ai/models/{model_id})")
api_key = os.getenv("api_key") or getpass.getpass("Training job Api Key:")
```

# API: Accessing the training job from python


```python
from brevettiai.platform import Job
 
job = Job.init(job_id=model_id, api_key=api_key)
```

    2022-02-02 15:52:43.493074: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-02-02 15:52:43.493110: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Config args found at: 's3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/info.json'


    kwargs <class 'NoneType'> not known


    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json


# Query the image samples in the training job


```python
from brevettiai.data.sample_tools import BrevettiDatasetSamples
samples = BrevettiDatasetSamples().get_image_samples(job.datasets)
samples.head(5)
```

    INFO:brevettiai.platform.dataset:Getting image samples from dataset 'NeurIPS vials TRAIN' [https://platform.brevetti.ai/data/cb14b6e3-b4b9-45bb-955f-47aa6489a192]
    INFO:brevettiai.platform.dataset:Getting annotations from dataset 'NeurIPS vials TRAIN' [https://platform.brevetti.ai/data/cb14b6e3-b4b9-45bb-955f-47aa6489a192] with filter: None
    INFO:brevettiai.platform.dataset:Contents: {('good',): 3, ('failed_cap',): 2, ('missing_cap',): 2}





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>folder</th>
      <th>path</th>
      <th>etag</th>
      <th>segmentation_path</th>
      <th>segmentation_etag</th>
      <th>bucket</th>
      <th>dataset</th>
      <th>dataset_id</th>
      <th>reference</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(failed_cap,)</td>
      <td>failed_cap</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>18082de95a667ad2b5c11c23deaf21c0</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>674b02943b722925696d1947d0cba9c3</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>NeurIPS vials TRAIN</td>
      <td>cb14b6e3-b4b9-45bb-955f-47aa6489a192</td>
      <td>N/A</td>
      <td>https://platform.brevetti.ai/download?path=cb1...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(failed_cap,)</td>
      <td>failed_cap</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>419fc5612ae56336d02e0f375f742dbe</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>2e9d79d3be01e983b67401ee1b967083</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>NeurIPS vials TRAIN</td>
      <td>cb14b6e3-b4b9-45bb-955f-47aa6489a192</td>
      <td>N/A</td>
      <td>https://platform.brevetti.ai/download?path=cb1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(good,)</td>
      <td>good</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>bb697e5c1a401385102c93f7e15d2827</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>d23abb9ea84d3d1b8ae7d2bb9fd47b5b</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>NeurIPS vials TRAIN</td>
      <td>cb14b6e3-b4b9-45bb-955f-47aa6489a192</td>
      <td>N/A</td>
      <td>https://platform.brevetti.ai/download?path=cb1...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(good,)</td>
      <td>good</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>b1df8d1fd453d87ff8b93477af1c0f9d</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>b50596b248f356955efb48434c0f60cc</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>NeurIPS vials TRAIN</td>
      <td>cb14b6e3-b4b9-45bb-955f-47aa6489a192</td>
      <td>N/A</td>
      <td>https://platform.brevetti.ai/download?path=cb1...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(good,)</td>
      <td>good</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>c4c3e8e083e7ef1b2a78b002e2dc6467</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>94be4aa63aaf6bc84cfdd1be96bb62d9</td>
      <td>s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f...</td>
      <td>NeurIPS vials TRAIN</td>
      <td>cb14b6e3-b4b9-45bb-955f-47aa6489a192</td>
      <td>N/A</td>
      <td>https://platform.brevetti.ai/download?path=cb1...</td>
    </tr>
  </tbody>
</table>
</div>



# Using job.io to download samples


```python
# io_tools is accessible from the job object or directly via import 'from brevettiai.io import io_tools'
# Note that access rights are configured on the IoTools object, and as such different instances of the object
# does not neccesarily have access to the same files. 
io_tools = job.io

# Buffer with binary file content of first image sample
buf = io_tools.read_file(samples.path[0])
buf[:10]
```




    b'BM6L\x02\x00\x00\x00\x00\x00'




```python
# Copy the first image sample file to your local drive
io_tools.copy(samples.path[0], samples.path[0].split("/")[-1])
print(f"s3 'bucket' path: {samples.path[0]}")
print("Listing local directory")
os.listdir(".")
```

    s3 'bucket' path: s3://data.criterion.ai/cb14b6e3-b4b9-45bb-955f-47aa6489a192.datasets.criterion.ai/failed_cap/0_1543413169486.bmp
    Listing local directory





    ['0_1543413169486.bmp', 'BrevettiAI_data_access.ipynb', 'image_classifier_101']



To explore the code by examples, please run the in the notebook that can be found on colab on this link [Brevettiai Data Access](https://githubtocolab.com/brevettiai/brevettiai-docs/blob/master/src/tutorials/BrevettiAI_data_access.ipynb)