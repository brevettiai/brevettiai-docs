# Brevetti AI package installation


```python
pip install -U git+https://bitbucket.org/criterionai/core
```

    Collecting git+https://bitbucket.org/criterionai/core
      Cloning https://bitbucket.org/criterionai/core to /tmp/pip-req-build-n_r_smy3
      Running command git clone -q https://bitbucket.org/criterionai/core /tmp/pip-req-build-n_r_smy3
    Collecting backoff>=1.10
      Downloading https://files.pythonhosted.org/packages/f0/32/c5dd4f4b0746e9ec05ace2a5045c1fc375ae67ee94355344ad6c7005fd87/backoff-1.10.0-py2.py3-none-any.whl
    Requirement already satisfied, skipping upgrade: numpy>=1.18 in /usr/local/lib/python3.6/dist-packages (from brevetti-ai==1.0) (1.18.5)
    Requirement already satisfied, skipping upgrade: pandas>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from brevetti-ai==1.0) (1.1.4)
    Collecting configparser
      Downloading https://files.pythonhosted.org/packages/08/b2/ef713e0e67f6e7ec7d59aea3ee78d05b39c15930057e724cc6d362a8c3bb/configparser-5.0.1-py3-none-any.whl
    Collecting minio>=5.0.10
    [?25l  Downloading https://files.pythonhosted.org/packages/36/fb/5f8f2768ae1a39e434abc570eac2f950770d64d6d714c914775b729fa507/minio-6.0.0-py2.py3-none-any.whl (72kB)
    [K     |████████████████████████████████| 81kB 3.9MB/s 
    [?25hRequirement already satisfied, skipping upgrade: requests>=2.23.0 in /usr/local/lib/python3.6/dist-packages (from brevetti-ai==1.0) (2.23.0)
    Requirement already satisfied, skipping upgrade: altair==4.1.0 in /usr/local/lib/python3.6/dist-packages (from brevetti-ai==1.0) (4.1.0)
    Requirement already satisfied, skipping upgrade: tqdm in /usr/local/lib/python3.6/dist-packages (from brevetti-ai==1.0) (4.41.1)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.22 in /usr/local/lib/python3.6/dist-packages (from brevetti-ai==1.0) (0.22.2.post1)
    Collecting plotly>=4.6.0
    [?25l  Downloading https://files.pythonhosted.org/packages/a6/66/af86e9d9bf1a3e4f2dabebeabd02a32e8ddf671a5d072b3af2b011efea99/plotly-4.12.0-py2.py3-none-any.whl (13.1MB)
    [K     |████████████████████████████████| 13.1MB 316kB/s 
    [?25hCollecting tensorflow-addons>=0.11.2
    [?25l  Downloading https://files.pythonhosted.org/packages/b3/f8/d6fca180c123f2851035c4493690662ebdad0849a9059d56035434bff5c9/tensorflow_addons-0.11.2-cp36-cp36m-manylinux2010_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 45.6MB/s 
    [?25hRequirement already satisfied, skipping upgrade: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=1.0.2->brevetti-ai==1.0) (2018.9)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /usr/local/lib/python3.6/dist-packages (from pandas>=1.0.2->brevetti-ai==1.0) (2.8.1)
    Requirement already satisfied, skipping upgrade: urllib3 in /usr/local/lib/python3.6/dist-packages (from minio>=5.0.10->brevetti-ai==1.0) (1.24.3)
    Requirement already satisfied, skipping upgrade: certifi in /usr/local/lib/python3.6/dist-packages (from minio>=5.0.10->brevetti-ai==1.0) (2020.6.20)
    Requirement already satisfied, skipping upgrade: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests>=2.23.0->brevetti-ai==1.0) (2.10)
    Requirement already satisfied, skipping upgrade: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests>=2.23.0->brevetti-ai==1.0) (3.0.4)
    Requirement already satisfied, skipping upgrade: jinja2 in /usr/local/lib/python3.6/dist-packages (from altair==4.1.0->brevetti-ai==1.0) (2.11.2)
    Requirement already satisfied, skipping upgrade: jsonschema in /usr/local/lib/python3.6/dist-packages (from altair==4.1.0->brevetti-ai==1.0) (2.6.0)
    Requirement already satisfied, skipping upgrade: toolz in /usr/local/lib/python3.6/dist-packages (from altair==4.1.0->brevetti-ai==1.0) (0.11.1)
    Requirement already satisfied, skipping upgrade: entrypoints in /usr/local/lib/python3.6/dist-packages (from altair==4.1.0->brevetti-ai==1.0) (0.3)
    Requirement already satisfied, skipping upgrade: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.22->brevetti-ai==1.0) (1.4.1)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.22->brevetti-ai==1.0) (0.17.0)
    Requirement already satisfied, skipping upgrade: six in /usr/local/lib/python3.6/dist-packages (from plotly>=4.6.0->brevetti-ai==1.0) (1.15.0)
    Requirement already satisfied, skipping upgrade: retrying>=1.3.3 in /usr/local/lib/python3.6/dist-packages (from plotly>=4.6.0->brevetti-ai==1.0) (1.3.3)
    Requirement already satisfied, skipping upgrade: typeguard>=2.7 in /usr/local/lib/python3.6/dist-packages (from tensorflow-addons>=0.11.2->brevetti-ai==1.0) (2.7.1)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2->altair==4.1.0->brevetti-ai==1.0) (1.1.1)
    Building wheels for collected packages: brevetti-ai
      Building wheel for brevetti-ai (setup.py) ... [?25l[?25hdone
      Created wheel for brevetti-ai: filename=brevetti_ai-1.0-cp36-none-any.whl size=74319 sha256=c234d4290180028c56cc732527a0192e6798bc7a3b8d60069072de536a7af5e3
      Stored in directory: /tmp/pip-ephem-wheel-cache-o7ejicm7/wheels/f2/b5/4d/7aa387de3df221d00e09a02515742b513e923b369c6b724a19
    Successfully built brevetti-ai
    Installing collected packages: backoff, configparser, minio, plotly, tensorflow-addons, brevetti-ai
      Found existing installation: plotly 4.4.1
        Uninstalling plotly-4.4.1:
          Successfully uninstalled plotly-4.4.1
      Found existing installation: tensorflow-addons 0.8.3
        Uninstalling tensorflow-addons-0.8.3:
          Successfully uninstalled tensorflow-addons-0.8.3
    Successfully installed backoff-1.10.0 brevetti-ai-1.0 configparser-5.0.1 minio-6.0.0 plotly-4.12.0 tensorflow-addons-0.11.2
    

# BrevettiAI Web API module examples
brevettiai platform web api is a lightweight api for interfacing with the brevettiai platform.

This notebook documents simple usage of it, in a development context through examples

This notebook illustrates the Brevetti AI web access

* High level access for automation of tasks on the platform, tagging, dataset management, models, etc...


Web access is granted with your website user, allowing you to automate tasks on the platform. In Python this is achieved through the **BrevettiAI** object.


```python
# Setup logging
import logging
log = logging.getLogger(__name__)
logging.basicConfig()
log.root.setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Imports and setup
from brevettiai.platform import BrevettiAI
```

# BrevettiAI Access API

High level access for automation of tasks on the platform, tagging, dataset management, models, etc...

## Platform Login
As on the web page you have 15 minutes of access before needing to log back in.


```python
web = BrevettiAI()
```

    https://platform.brevetti.ai - username: yhns@novonordisk.com
    Password:··········
    


```python
help(web)
```

# List datasets, tags, models...

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




    ['Kaluga CVT25 2019-05-21 Detemir JJ31529 Plunger',
     '24-5095910-CC01 ccbottom transformed',
     'CVT24 JW55N89 Accepted Particle 1',
     'Agar plates: BacSpot test 2/Sep 17 lenw dark',
     'Ryzodec test kit : particle defects (Particle 2) ',
     'CVT24_CAP_DEVNewCapCrimper_NoAir_LowLight',
     'CVT24 JW55R87 Accepted Cap',
     'Contentkit 5091000 BottomCC',
     'CVT24 5026903 JW56W79 Accepted Particle 1 penmix50',
     'CC-bottom: ImageLog_gain60_24-5190232-CC02']



For a single dataset, model or ... use the get_... functions with an id


```python
dataset = web.get_dataset(datasets[0]["id"])
dataset
```




    {'created': '2019-07-17T21:29:11.528757',
     'folderName': '004432c5-5b70-4b65-a1ea-0a790f4b85cd.datasets.criterion.ai',
     'folders': [],
     'id': '004432c5-5b70-4b65-a1ea-0a790f4b85cd',
     'locked': False,
     'name': 'Kaluga CVT25 2019-05-21 Detemir JJ31529 Plunger',
     'notes': None,
     'reference': 'JJ31529',
     'tagIds': ['2091bee9-24e3-4bde-9f23-23dd6cb77595',
      'a4510257-d135-4dad-90e9-255cfe6b43bb',
      '314a61ca-e435-4a17-bd33-12887c67944b',
      '952c71bc-0f0e-496d-8d76-53de56c16a17',
      '4d925162-abfa-4dfc-b07a-c82e142604e5']}



## Create Job
To enter the job context you can either create a model on the platform or programatically via the web api.

The following code finds the firs dataset and creates a model (job) with access to this model.
The model context type is the id of a model type on the platform to use.
After running the model is available on the website, along with an s3 bucket for artifacts for your job outputs


When creating a model you have the option to include datasets and tags and settings defining your model.


```python
# Datasets to add to the created job
datasets = web.get_dataset()[1:2]

# Settings used for the job
settings = dict(some_setting_i_need=1)

model_context_type = "a0aaad69-c032-41c1-a68c-e9a15a5fb18c" # "Magic" undocumented uId of *external* job model type

model_def = web.create_model(name=f'Test {web.user["firstName"]} {web.user["lastName"]}',
                             model_type=model_context_type,
                             settings=settings,
                             datasets=datasets)
```

## Start job

The model id and the model api key gives you access to use the python sdk to access data, and to upload artifacts and lots of other cool stuff. To enable this, we need to start model training - this is the same as selecting "Simulate training" on the platform.


```python
# Starting training in simulate mode
web.start_model_training(model=model_def['id'])
print(f"Model url: {web.baseurl}/models/{model_def['id']} (Please check it out :)\n")
print("To access data and model through python SDK use the following")
print(f"Model id: {model_def['id']}")
print(f"Model api key: {model_def['apiKey']}")
```
