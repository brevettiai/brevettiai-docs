---
description: >-
  This section aims to describe the use of the Job API, through a number of
  guides
---

# Job API

The Job API is your execution context when running a job on the platform. A job being defined as a model training process, or the process of creating a test report.

It provides a python interface to the website, and keeps track of the resources you have available there, and parsing of input in the form of settings to your code.

![](BrevettiAI_Job_API.png)

From the platform, a job \(model or test report\) is an input configuration with a storage space attached. The storage space freely available to use by the job, but following a few conventions, allows the platform to parse specific content for display on the model page.

## Job lifecycle

### Start

To begin executing a job you first need do get an execution context, thereby start the job. To do this you run the application.init function. This returns a CriterionConfig object to you, which is your configuration for the job.

```text
from brevettiai.platform import Job
job = Job.init()
```

This command expects the job GUID and API key as arguments via argv or parameters on the function.

Settings may be added to the job by creating a settings definition \(schema\). This schema is parsed to generate UI on the platform, and parsed by the config object, to use specific modules. Pass a `path` to a schema or a `SchemaBuilder` object to the `init` function to use apply it to the configuration.

### Execute

Run your application code, and use tooling for integration with the platform features.

Add custom output metrics to your job.

```text
print(f"Uploading metrics and outputs to {job.host_name}")
job.add_output_metric("My custom metric", 277)
job.upload_job_output()
```

### Complete

On completion, complete the job on the platform to signal that you are done with the job. This action is performed by calling `complete_job` on the config object you got when starting the job. You can additionally call upload\_job\_output, to serialize the configuration object and upload metrics added during the training. 

```text
job.complete_job(path_to_artifact_with_model_archive)
```

