#  Accessing the training job from python


```python
from brevettiai.platform import Job
 
job = Job.init(job_id=model_id, api_key=api_key)
```

    2022-02-02 11:30:39.989742: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-02-02 11:30:39.989778: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


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
    INFO:brevettiai.platform.dataset:Contents: {('good',): 3, ('missing_cap',): 2, ('failed_cap',): 2}





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


