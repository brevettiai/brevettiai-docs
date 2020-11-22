# Interfaces

Models and reports support a number of different visualization types on the platform.

## Pivot tables

Use pivot tables to get an overview over the contents of your job. Export pivot table information files to the pivot directory of the artifact dir.

```text
from brevettiai.interfaces.pivot import export_pivot_table, get_default_fields, pivot_fields
export_pivot_table(job.artifact_path("pivot", dir=True), samples,
                   datasets=job.datasets,
                   fields=None,
                   tags=job.get_root_tags(),
                   rows=["dataset_id"],
                   cols=["category"],
                   agg={"url": "first"})
```

## Progress monitoring \(Models only\)

Add progress metrics to monitor your models while it is running, by adding the RemoteMonitor callback to your keras training loop or call it yourself in your training code.

```text
from brevettiai.interfaces.remote_monitor import RemoteMonitor
callback = RemoteMonitor(root=job.host_name, path=job.api_endpoints["remote"])
```

## HTML

Add your own custom html reports. All HTML files present in the root of the artifacts directory will be shown with the model and test reports.

## Facets Dive

Use [Facets Dive](https://pair-code.github.io/facets/) to explore your data interactively.

![Example of Facets Dive tool](../../../.gitbook/assets/image.png)

Use the facets implementation tool with the image datasets to generate your own facets charts, and put the files in the facets folder in your artifacts.

To use the built-in tools you need to supply a DataGenerator which outputs a 64x64 thumbnail image, and category.

```text
from brevettiai.data.data_generator import DataGenerator
from brevettiai.interfaces.facets_atlas import build_facets

fds = DataGenerator(samples, shuffle=True, output_structure=("img", "category")) \
    .map(ImagePipeline(target_size=(64,64), antialias=True))

build_facets(fds, job.artifact_path("facets", dir=True), count=32)
```

## Vega-Lite charts

Add [Vega-Lite charts](https://vega.github.io/vega-lite/) to your model page by calling upload\_chart on the configuration object. Some different standard charts are available under `brevettiai.interfaces.vegalite_charts`

```text
from brevettiai.interfaces import vegalite_charts

vegalite_json = vegalite_charts.dataset_summary(samples)
job.upload_chart(vegalite_json)
```

