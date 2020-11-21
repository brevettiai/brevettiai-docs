# Model export

Export your model to an archive. ex a tar.gz zipped tensorflow saved\_model. Place this model in the artifact path, and include the path in the job completion call

```text
job.complete_job(path_to_artifact_with_model_archive)
```

