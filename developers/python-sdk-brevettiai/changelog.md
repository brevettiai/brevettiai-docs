---
description: >-
  This page presents an overview of all changes to the Brevetti AI Platform
  Python SDK
---

# Changelog

## 1.0

* Cleanup of core apis
  * Rename of package from criterion\_core to brevettiai
  * Move platform interface code to brevettiai.platform
  * Move criterion\_core.applications to brevettiai.interfaces containing interfaces to external elements
  * Renaming of CriterionConfig to Job
  * Renaming of TfDataset to DataGenerator
* New settings API
  * extend vue\_schema\_utils.VueSettingsModule to get a settings object with a to\_schema function for schema generation.

