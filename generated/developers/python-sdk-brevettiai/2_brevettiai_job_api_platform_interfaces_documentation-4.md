#  Complete job
Update the following on the platform
* The path to the model file (optional)
* That the training or testing process is finished, so the UI can be updated
* This revokes access to write to the job artifact path, and access to the datasets

## Model export

Export your model to an archive. ex a tar.gz zipped tensorflow saved\_model. Place this model in the artifact path, and include the path in the job completion call


```python
# job.complete_job(exported_model_path)
```
