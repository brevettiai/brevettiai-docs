# Sample split - sample integrity module

Functionality to split samples between training and development sets \(often referred to as validation set, but this name is confusing in a regulated environment\)

This module allows for more fine grained control of the splitting process than what is provided by e.g. sklearn. The main feature is that it can split based on unique samples rather than just randomly. This is important when multiple images of the same physical item are available

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

```text
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
```

|  |  | category | path | etag | bucket | dataset | dataset\_id | url |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| folder | purpose |  |  |  |  |  |  |  |
| bad\_cap | devel | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| train | 16 | 16 | 16 | 16 | 16 | 16 | 16 |  |
| good | devel | 28 | 28 | 28 | 28 | 28 | 28 | 28 |
| train | 118 | 118 | 118 | 118 | 118 | 118 | 118 |  |
| missing\_cap | devel | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| train | 10 | 10 | 10 | 10 | 10 | 10 | 10 |  |

