#  Data generator and stratified sampler
The brevettiai DataGenerator object is a generator object that extents the functionality of tensorflow datasets by adding
* a generated random seed to the map function, so that an image augmentation pipeline may produce reproducible results
* the possibility for stratified sampling such that samples can be drawn with controlled freqeuncy from different groups of the dataset

the method get_dataset() returns a tensorflow dataset object with the above mentioned properties


```python
from brevettiai.data.data_generator import StratifiedSampler, DataGenerator

batch_size = 4
# creating a data generator with stratification across a grouping on "folder" and with a weight determined by the square root of number of samples
generator = StratifiedSampler(batch_size=batch_size, groupby=["folder"], group_weighing="square root", seed=1)\
        .get(samples, shuffle=True, repeat=True)

for sample in generator.get_dataset().take(2):
    print(sample["path"])
```

    tf.Tensor(
    [b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/2_1543412100881.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/44_1543412166931.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/missing_cap/1_1543412754274.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/missing_cap/4_1543412678089.bmp'], shape=(4,), dtype=string)
    tf.Tensor(
    [b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/24_1543412105328.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/good/18_1543412104125.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/bad_cap/9_1543413268396.bmp'
     b's3://public.data.criterion.ai/data/NeurIPS_2018_reduced/missing_cap/3_1543412754682.bmp'], shape=(4,), dtype=string)


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
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>good</th>
      <td>112</td>
      <td>112</td>
      <td>112</td>
      <td>112</td>
      <td>112</td>
      <td>112</td>
      <td>112</td>
      <td>112</td>
      <td>112</td>
    </tr>
    <tr>
      <th>missing_cap</th>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>


