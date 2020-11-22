# Interfaces / integrations

## Interfaces / integrations

### Job output to platform website

A number of different outputs are available on the platform, here is a subset.

### Metrics

Metrics which may be compared across models can be added via the config object.

```python
print(f"Uploading metrics and outputs to {job.host_name}/models/{model_id}/artifacts")
job.add_output_metric("My custom metric", 277)
job.upload_job_output()
```

```text
INFO:brevettiai.platform.job:Uploading output.json to s3://data.criterion.ai/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts/output.json


Uploading metrics and outputs to https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71/artifacts





'artifacts/output.json'
```

### Remote monitoring

```python
from brevettiai.interfaces.remote_monitor import RemoteMonitor
remote_monitor_callback = RemoteMonitor(root=job.host_name, path=job.api_endpoints["remote"])
# Simulate training epochs and produce callbacks
remote_monitor_callback.on_epoch_end(11, {"loss": 0.9, "accuracy": 0.5})
remote_monitor_callback.on_epoch_end(12, {"loss": 0.7, "accuracy": 0.8})

print(f"Training progress visible on {job.host_name}/models/{model_id}")
```

```text
Training progress visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71
```

### Pivot tables

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

```text
Pivot table visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71
```

### Facets

Create facet dives to explore your data in depth by creating a dataset outputting thumbnails of size \(64x64\) per sample

```python
from brevettiai.interfaces.facets_atlas import build_facets
from brevettiai.data.data_generator import StratifiedSampler, DataGenerator
fds = DataGenerator(samples, shuffle=True, output_structure=("img", "category")) \
    .map(ImagePipeline(target_size=(64,64), antialias=True))

build_facets(fds, job.artifact_path("facets", dir=True), count=32)

print(f"Facets visible on {job.host_name}/models/{model_id}")
```

```text
Facets visible on https://platform.brevetti.ai/models/ae03ff72-b7a2-444d-8fe9-623f61dc4c71
```

## Complete job to update the following on the platform

* The path to the model file \(optional\)
* That the training or testing process is finished, so the UI can be updated
* That access to write to the artifact path, and access to the datasets should no longer be granted

```python
# job.complete_job()
```

