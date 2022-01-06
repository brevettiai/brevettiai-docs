#  Customized Job Settings
Settings are essentially the serialized configuration of a training job algorithm.
Settings can be used for a training job configuration by letting a user change settings, and settings are included in the default job output, such that the parameters of a training job can be saved and tracked for comparison and audit purposes.


```python
from brevettiai.interfaces.vue_schema_utils import VueSettingsModule as SettingsModule

class MyAlgoObject(SettingsModule):
    def __init__(self, multiply_factor: float = 2.0, 
                 enable : bool = True):
        self.multiply_factor = multiply_factor
        self.enable = enable
    def __call__(self, x):
        factor = 1.0
        if self.enable:
            factor *= self.multiply_factor
        return x * factor
test_obj = MyAlgoObject(multiply_factor=3.0)

# Settings used for creating the job
settings = test_obj.get_settings()
print(settings)
```

    {'multiply_factor': 3.0, 'enable': True}

