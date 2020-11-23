---
description:
  This section shows you how to deploy code to the model zoo, to be available
  for other users in your organization.
---

# Deployment

Job code can be deployed to the platform model zoo, for others in the organization to use. In general terms the code is packaged to a settings schema and a Docker container. The schema is used to generate UI for settings selection on the platform. When the job is started the Docker container is run on a virtual server with drivers and hardware to support training of deep learning models.

## Building a docker container

Containers running on the platform have few requirements.

* The entry point must start the training / test report
* The container application must support the following arguments
  * model\_id / test\_id; GUID for the job
  * api\_key; Job access key
  * info\_file; Settings for job
  * job\_dir; location of job persistent storage directory



Start from the **Telescope** example application

[Telescope application bitbucket repository](https://bitbucket.org/criterionai/telescope/src/master/)



