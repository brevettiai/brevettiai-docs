---
description: >-
  This page presents an overview of all changes to the Brevetti AI Platform
  Python SDK
---

# Changelog

## 0.4.1
* Introduce cache path environment variable: `BREVETTI_AI_CACHE`
* Move cache path 1 level up (`[cache_path]/cache/s3` -> `[cache_path]/s3`)
* Add Tensorboard collector tooling brevettiai `poetry run python -m brevettiai.utils.tensorboard_collector -h`
* Allow ImageProcessor to change shape and channel numbers
* Minor bug fixes



