# Sampling

## List samples

Samples can be listed directly from the CriterionDataset via `get_image_samples` or the `sample_tools.get_image_samples` functions for multiple datasets.

Annotations from the platform may also be merged with the annotations keyword as a boolean, or a dict for special annotation merging parameters.

```text
from criterion_core.data import get_image_samples
get_image_samples(config.datasets, 
    settings.data.class_mapping, force_categories=classes_list,
    annotations=True)
```

## Sample preparation

Use the sample preparation tool to manage your samples and assign them into test / development / train. use the artifact sample\_identification.csv file as good practice to store the results, and use for initial state input.

```text
from criterion_core.data import sample_integrity
purpose_args = dict() # arguments for assign_sample_purpose
sample_integrity.sample_preparation(
        full_set, id_path=config.artifact_path("sample_identification.csv"),
        purpose_args=purpose_args)
```

## Sample Integrity

Criterion core uses AWS etags \(Usually MD5 checksums\), and file MD5 checksums as method of sample integrity checking. This allows fast listing of identities via the s3 list bucket api for most object, and s3 file metadata storage for the rest.

With the MD5 checksums it is possible to alert the user to duplicate samples, and to ensure that duplicates are used for the same purpose \(training/development/test\).

## Purpose assignment

Criterion core splits the samples of the job into multiple groups based on a strategy, while keeping track of the sample md5 identity. The purpose assignment support sharding and stratification, to split the data based on information on the samples.

