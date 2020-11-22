# Brevetti AI Dataset

The dataset object can be used to manage listing of data \(and access, if it where not publicly available\)

```python
from brevettiai.platform.dataset import Dataset
ds = Dataset(bucket=dataset_path, resolve_access_rights=False)

# Fix to get access to a public bucket without credentials
ds.io.minio.client_factory("s3://public.data.criterion.ai", lambda **x:{"endpoint": "s3-eu-west-1.amazonaws.com"})

samples = ds.get_image_samples()
# Printing content of a sample from the pandas data frame
print("Sample: ", samples.sample(1).iloc[0].to_dict())
```

```text
Sample:  {'category': ('good',), 'folder': 'good', 'path': 's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/28_1543412163686.bmp', 'etag': 'b180108423a8469715a7d1200377ac0e', 'bucket': 's3://public.data.criterion.ai/data/NeurIPS_2018_reduced', 'dataset': '', 'dataset_id': '', 'url': 'https://platform.brevetti.ai/download?path=lic.data.criterion.ai%2Fdata%2FNeurIPS_2018_reduced%2Fgood%2F28_1543412163686.bmp'}
```

Samples now holds the image samples in a pandas dataframe object. We can e.g. investigate the distribution of the different classes

```python
samples.groupby("folder").count()
```

|  | category | path | etag | bucket | dataset | dataset\_id | url |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| folder |  |  |  |  |  |  |  |
| bad\_cap | 22 | 22 | 22 | 22 | 22 | 22 | 22 |
| good | 146 | 146 | 146 | 146 | 146 | 146 | 146 |
| missing\_cap | 12 | 12 | 12 | 12 | 12 | 12 | 12 |

