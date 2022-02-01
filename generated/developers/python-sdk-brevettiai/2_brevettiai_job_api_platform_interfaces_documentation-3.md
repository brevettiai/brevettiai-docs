#  Interfaces / integrations
##Job output to platform website
A number of different outputs are available on the platform, here is a subset.

## Metrics
Metrics which may be compared across models can be added via the config object.


```python
print(f"Uploading metrics and outputs to {job.host_name}/models/{model_id}/artifacts")
job.add_output_metric("My custom metric", 277)
job.upload_job_output()
```

    INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json


    Uploading metrics and outputs to https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts





    'artifacts/output.json'



## Progress monitoring (Models only)
Add progress metrics to monitor your models while it is running, by adding the RemoteMonitor callback to your keras training loop or call it yourself in your training code.


```python
from brevettiai.interfaces.remote_monitor import RemoteMonitor
remote_monitor_callback = RemoteMonitor(root=job.host_name, path=job.api_endpoints["remote"])
# Simulate training epochs and produce callbacks
remote_monitor_callback.on_epoch_end(11, {"loss": 0.9, "accuracy": 0.5})
remote_monitor_callback.on_epoch_end(12, {"loss": 0.7, "accuracy": 0.8})

print(f"Training progress visible on {job.host_name}/models/{model_id}")
```

    Training progress visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71


## Pivot tables

create pivot tables on the web platform to get an overview over your data


```python
from brevettiai.interfaces.pivot import export_pivot_table, get_default_fields, pivot_fields
export_pivot_table(job.artifact_path("pivot", dir=True), samples,
                   datasets=job.datasets,
                   fields=None,
                   tags=job.get_root_tags(),
                   rows=["dataset_id"],
                   cols=["category"],
                   agg={"url": "first"})
print(f"Pivot table visible on {job.host_name}/models/{model_id}")
```

    Pivot table visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71


## Facets
Create facet dives to explore your data in depth by creating a dataset outputting thumbnails of size (64x64) per sample. 
![Facet example](https://gblobscdn.gitbook.com/assets%2F-LY12YhLSCDWlqNaQqWT%2F-MIdFH6dqJxgrYtQH83E%2F-MIdJ3qn1kPxLh6K0YI0%2Fimage.png?alt=media&token=d59993dc-9dd0-4f97-a548-4d6ceddf257d)

Put the files in the facets folder in your artifacts. To use the built-in tools you need to supply a DataGenerator which outputs a 64x64 thumbnail image, and category.


```python
from brevettiai.interfaces.facets_atlas import build_facets
from brevettiai.data.data_generator import StratifiedSampler, DataGenerator
fds = DataGenerator(samples, shuffle=True, output_structure=("img", "category")) \
    .map(ImagePipeline(target_size=(64,64), antialias=True))

build_facets(fds, job.artifact_path("facets", dir=True), count=32)

print(f"Facets visible on {job.host_name}/models/{model_id}")

build_facets(fds, job.artifact_path("facets", dir=True), count=32)
```

    100%|██████████| 32/32 [00:02<00:00, 11.32it/s]


    Facets visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71


    100%|██████████| 32/32 [00:00<00:00, 560.91it/s]





    True



## Vega-lite charts
Vega-Lite charts
Add Vega-Lite charts to your model page by calling upload_chart on the configuration object. Some different standard charts are available under brevettiai.interfaces.vegalite_charts


```python
from brevettiai.interfaces import vegalite_charts

vegalite_json = vegalite_charts.dataset_summary(samples)
job.upload_chart("demo", vegalite_json)
```




    <Response [204]>


