#  Brevetti AI Dataset
The dataset object can be used to manage listing of data (and access, if it where not publicly available)


```python
from brevettiai.platform.dataset import Dataset
ds = Dataset(bucket=dataset_path, resolve_access_rights=False)

# Fix to get access to a public bucket without credentials
ds.io.minio.client_factory("s3://public.data.criterion.ai", lambda **x:{"endpoint": "s3-eu-west-1.amazonaws.com"})

samples = ds.get_image_samples()
# Printing content of a sample from the pandas data frame
print("Sample: ", samples.sample(1).iloc[0].to_dict())
```

    CV2 not available


    Sample:  {'category': ('good',), 'folder': 'good', 'path': 's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/19_1543412161923.bmp', 'etag': '8563d97a8a3bad9329551655dd4ca154', 'bucket': 's3://public.data.criterion.ai/data/NeurIPS_2018_reduced', 'dataset': '', 'dataset_id': '', 'url': 'https://platform.brevetti.ai/download?path=lic.data.criterion.ai%2Fdata%2FNeurIPS_2018_reduced%2Fgood%2F19_1543412161923.bmp'}


Samples now holds the image samples in a pandas dataframe object. We can e.g. investigate the distribution of the different classes


```python
samples.groupby("folder").count()
```




<div>

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


