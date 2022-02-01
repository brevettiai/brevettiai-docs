#  Settings
The *brevettiai* library contains an interface to easily serialize and deserialize objects in json format. This has several advantages in designing image computer vision pipelines
* It is easy to alter parameters when running new training jobs / experiments
* The training process parameters can be stored for each experiment to keep track of experiments

Deriving classes from <code>brevettiai.interfaces.vue_schema_utils.VueSettingsModule</code> will enable <code>get_config</code> and <code>from_config</code> methods to manage serialization. Parameters should be specified with type hints, and the initializer parameter names should match the member variable names.


```python
# Job info: NB: replace with ID and api key from your job
from brevettiai.interfaces.vue_schema_utils import VueSettingsModule


# NB: all parameters are asigned as member variables
class MyCustomObject(VueSettingsModule):
    def __init__(self, custom_param: int = 0):
        self.custom_param = custom_param

# NB: all parameters are asigned as member variables
class SerializeableObject(VueSettingsModule):
    def __init__(self, param_1: float = -1.0, param_2: str = None, param_3: dict = None, custom: MyCustomObject = None):
        self.param_1 = param_1
        self.param_2 = param_2 or "Param2 not provided"
        self.param_3 = param_3 or {"nested_param"}
        self.custom = custom or MyCustomObject()

my_object = SerializeableObject(param_1=1, param_2="Test", param_3={"test_dict_serialization": "nested"})
# Serialize objects with get_config
config = my_object.get_config()

print("Original object config: \n", config)

# Deserialize objects with from_config
my_object_copy = SerializeableObject.from_config(config)

print("Deserialized object config: \n", my_object_copy.get_config())
```

    2022-02-01 09:53:42.082615: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.7.12/x64/lib
    2022-02-01 09:53:42.082654: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    DEBUG:tensorflow:Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud-tpu-client.
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7
    DEBUG:h5py._conv:Creating converter from 7 to 5
    DEBUG:h5py._conv:Creating converter from 5 to 7


    Original object config: 
     {'param_1': 1, 'param_2': 'Test', 'param_3': {'test_dict_serialization': 'nested'}, 'custom': {'custom_param': 0}}
    Deserialized object config: 
     {'param_1': 1, 'param_2': 'Test', 'param_3': {'test_dict_serialization': 'nested'}, 'custom': {'custom_param': 0}}

