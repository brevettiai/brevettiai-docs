#  Sample split - sample integrity module
Functionality to split samples between training and development sets (often referred to as validation set, but this name is confusing in a regulated environment)

This module allows for more fine grained control of the splitting process than what is provided by e.g. sklearn.
The main feature is that it can split based on unique samples rather than just randomly. This is important when multiple images of the same physical item are available


```python
from brevettiai.data.sample_integrity import SampleSplit
from IPython.display import display 

uniqueness_regex = r"/(\d*)_\d*.bmp"

samples = SampleSplit(stratification=["folder"], uniqueness=uniqueness_regex, split=0.8, seed=42).assign(samples, remainder="devel")
print("Devel samples")
display(samples[samples["purpose"] == "devel"][:5].path.values)
print("Train samples")
display(samples[samples["purpose"] == "train"][:5].path.values)

samples.groupby(["folder", "purpose"]).count()
```

    Devel samples



    array(['s3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/2_1543413180595.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/2_1543413190226.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/5_1543413258015.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/5_1543413267597.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/6_1543413181402.bmp'],
          dtype=object)


    Train samples



    array(['s3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/0_1543413169486.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/0_1543413189854.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/10_1543413182213.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/10_1543413191789.bmp',
           's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/1_1543413257224.bmp'],
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
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>train</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">good</th>
      <th>devel</th>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
    </tr>
    <tr>
      <th>train</th>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
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

