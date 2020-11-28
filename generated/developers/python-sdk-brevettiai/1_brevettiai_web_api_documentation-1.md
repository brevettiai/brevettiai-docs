#  Element acces, list datasets, tags, models...

with the web object you can list, manage and edit elements on the web page.
Most of the functions require an id, the guid of the object to manipulate. Alternatively to get all use None as id.

EX: to list datasets, tags, and models, call get_... with no id (id=None)


```python
datasets = web.get_dataset()
tags = web.get_tag()
models = web.get_model()

# List 10 first dataset names
[d["name"] for d in datasets][:10]
```




    ['NeurIPS 2018',
     'NeurIPS 2018 large',
     'Blood Cell Images',
     'Agar plates',
     'NeurIPS vials TRAIN']



For a single dataset, model or ... use the get_... functions with an id


```python
dataset = web.get_dataset(datasets[0]["id"])
dataset
```




    {'id': '21263db3-1c9b-456b-be1d-ecfa2afb5d99',
     'created': '2019-01-17T16:07:44.307313',
     'name': 'NeurIPS 2018',
     'reference': 'Batch HW0001',
     'notes': '',
     'folderName': '21263db3-1c9b-456b-be1d-ecfa2afb5d99.datasets.criterion.ai',
     'locked': False,
     'folders': [],
     'tagIds': []}


