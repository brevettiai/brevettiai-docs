description: |
    API documentation for modules: brevettiai, brevettiai.data, brevettiai.data.data_generator, brevettiai.data.image, brevettiai.data.image.annotation, brevettiai.data.image.annotation_loader, brevettiai.data.image.annotation_parser, brevettiai.data.image.annotation_pooling, brevettiai.data.image.bayer_demosaic, brevettiai.data.image.feature_calculator, brevettiai.data.image.image_augmenter, brevettiai.data.image.image_loader, brevettiai.data.image.image_pipeline, brevettiai.data.image.image_processor, brevettiai.data.image.multi_frame_imager, brevettiai.data.image.segmentation_loader, brevettiai.data.image.utils, brevettiai.data.sample_integrity, brevettiai.data.sample_tools, brevettiai.data.tf_utils, brevettiai.interfaces, brevettiai.interfaces.aws, brevettiai.interfaces.facets_atlas, brevettiai.interfaces.pivot, brevettiai.interfaces.raygun, brevettiai.interfaces.remote_monitor, brevettiai.interfaces.sagemaker, brevettiai.interfaces.vegalite_charts, brevettiai.interfaces.vue_schema_utils, brevettiai.io, brevettiai.io.credentials, brevettiai.io.h5_metadata, brevettiai.io.local_io, brevettiai.io.minio_io, brevettiai.io.onnx, brevettiai.io.openvino, brevettiai.io.path, brevettiai.io.serialization, brevettiai.io.tf_recorder, brevettiai.io.utils, brevettiai.model, brevettiai.model.factory, brevettiai.model.factory.lraspp, brevettiai.model.factory.mobilenetv2_backbone, brevettiai.model.factory.segmentation, brevettiai.model.losses, brevettiai.model.metadata, brevettiai.model.metadata.image_segmentation, brevettiai.model.metadata.metadata, brevettiai.platform, brevettiai.platform.models, brevettiai.platform.models.annotation, brevettiai.platform.models.dataset, brevettiai.platform.models.job, brevettiai.platform.models.platform_backend, brevettiai.platform.models.tag, brevettiai.platform.models.web_api_types, brevettiai.platform.platform_credentials, brevettiai.platform.web_api, brevettiai.tests, brevettiai.tests.test_data, brevettiai.tests.test_data_image, brevettiai.tests.test_data_manipulation, brevettiai.tests.test_image_loader, brevettiai.tests.test_model_metadata, brevettiai.tests.test_pivot, brevettiai.tests.test_platform_job, brevettiai.tests.test_polygon_extraction, brevettiai.tests.test_schema, brevettiai.tests.test_tags, brevettiai.utils, brevettiai.utils.argparse_utils, brevettiai.utils.dict_utils, brevettiai.utils.model_version, brevettiai.utils.module, brevettiai.utils.numpy_json_encoder, brevettiai.utils.pandas_utils, brevettiai.utils.polygon_utils, brevettiai.utils.profiling, brevettiai.utils.singleton, brevettiai.utils.tag_utils, brevettiai.utils.tf_serving_request, brevettiai.utils.validate_args.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...



# Module `brevettiai` {#id}





## Sub-modules

* [brevettiai.data](#brevettiai.data)
* [brevettiai.interfaces](#brevettiai.interfaces)
* [brevettiai.io](#brevettiai.io)
* [brevettiai.model](#brevettiai.model)
* [brevettiai.platform](#brevettiai.platform)
* [brevettiai.tests](#brevettiai.tests)
* [brevettiai.utils](#brevettiai.utils)







# Module `brevettiai.data` {#id}





## Sub-modules

* [brevettiai.data.data_generator](#brevettiai.data.data_generator)
* [brevettiai.data.image](#brevettiai.data.image)
* [brevettiai.data.sample_integrity](#brevettiai.data.sample_integrity)
* [brevettiai.data.sample_tools](#brevettiai.data.sample_tools)
* [brevettiai.data.tf_utils](#brevettiai.data.tf_utils)







# Module `brevettiai.data.data_generator` {#id}







## Functions



### Function `build_dataset_from_samples` {#id}




>     def build_dataset_from_samples(
>         samples,
>         groupby='category',
>         weighing='uniform',
>         shuffle=True,
>         repeat=True,
>         seed=None
>     )


Build tensorflow dataset from pandas dataframe with oversampling of groups
:param samples:
:param groupby:
:param weighing:
:param shuffle:
:param repeat:
:param seed: seed or np.random.RandomState
:return:


### Function `build_image_data_generator` {#id}




>     def build_image_data_generator(
>         samples,
>         classes=None,
>         image=None,
>         augmentation=None,
>         *args,
>         **kwargs
>     )


Utility function for building a default image dataset with images at "path" and class definitions at "category"
outputting image and onehot encoded class
:param samples: Pandas dataframe of samples, with at least columns (path, category)
:param classes: list of classes or none to autodetect from samples
:param image: kwargs for ImageLoader
:param augmentation: kwargs for ImageAugmenter
:param args: args for TfDataset
:param kwargs: args for TfDataset
:return: (image, onehot)


### Function `get_dataset` {#id}




>     def get_dataset(
>         df,
>         shuffle,
>         repeat,
>         seed=None
>     )


Build simple tensorflow dataset from pandas dataframe
:param df:
:param shuffle:
:param repeat:
:param seed: seed or np.random.RandomState
:return:


### Function `item_mapping` {#id}




>     def item_mapping(
>         df
>     )





### Function `map_output_structure` {#id}




>     def map_output_structure(
>         x,
>         structure
>     )





### Function `parse_weighing` {#id}




>     def parse_weighing(
>         weighing
>     )





### Function `predict_dataset` {#id}




>     def predict_dataset(
>         model,
>         dataset,
>         map_output=None
>     )


Predict results of model given dataset
:param model:
:param dataset:
:param map_output:
:return:


### Function `weighted_dataset_selector` {#id}




>     def weighted_dataset_selector(
>         weight
>     )






## Classes



### Class `DataGenerator` {#id}




>     class DataGenerator(
>         samples,
>         batch_size: int = 32,
>         shuffle: bool = False,
>         repeat: bool = False,
>         sampling_groupby: str = None,
>         sampling_group_weighing: str = 'uniform',
>         seed: int = None,
>         output_structure: tuple = None,
>         max_epoch_samples: int = inf
>     )


Dataset helper based on Tensorflow datasets, capable of seeding, weighted sampling, and tracking datasets for
logs.
:param samples: Pandas dataframe with inputs
:param batch_size: Number of samples per batch
:param shuffle: Shuffle items in dataset
:param repeat: Repeat samples from dataset
:param sampling_groupby: Stratified sample columns to group by when weighing each sample group for sampling
:param sampling_group_weighing: Stratfied sampling weighing function to use for weighing the sample groups supply function or select from ["uniform", "count", "square root", "log"]
:param seed: Seeding of dataset
:param output_structure: default output structure (tuples with keys) of dataset or None for full dictionary
:param max_epoch_samples: Max number of samples per epoch








#### Methods



##### Method `apply_tfds_actions` {#id}




>     def apply_tfds_actions(
>         self,
>         tfds
>     )





##### Method `build_dataset` {#id}




>     def build_dataset(
>         self,
>         ds: tensorflow.python.data.ops.dataset_ops.DatasetV2
>     ) ‑> tensorflow.python.data.ops.dataset_ops.DatasetV2


Extend this function to apply special functions to the dataset


##### Method `get_dataset` {#id}




>     def get_dataset(
>         self,
>         batch=True,
>         structure='__default__'
>     ) ‑> tensorflow.python.data.ops.dataset_ops.DatasetV2


Get tensorflow dataset
:param batch: output batched
:param structure: Structure of output, (None, "__default__" or structure of keys)
:return:


##### Method `get_dataset_actions` {#id}




>     def get_dataset_actions(
>         self
>     )


Get variable actions performed on datasets.

:return: list of actions, each action consisting of (callable,
args for callable, and args for tensorflow dataset map)


##### Method `get_dataset_numpy` {#id}




>     def get_dataset_numpy(
>         self,
>         *args,
>         **kwargs
>     )


Get numpy iterator of Dataset, similar interface as .get_dataset()
:return:


##### Method `get_debug_info` {#id}




>     def get_debug_info(
>         self
>     )





##### Method `get_samples` {#id}




>     def get_samples(
>         self,
>         batch=True,
>         structure=None
>     ) ‑> tensorflow.python.data.ops.dataset_ops.DatasetV2


Get tensorflow samples as tensorflow dataset without applying maps for e.g. loading data
:param batch: output batched
:param structure: Structure of output, (None, "__default__" or structure of keys)
:return:


##### Method `get_samples_numpy` {#id}




>     def get_samples_numpy(
>         self,
>         *args,
>         **kwargs
>     )


Get numpy iterator of samples in dataset, similar interface as .get_samples()
:return:


##### Method `map` {#id}




>     def map(
>         self,
>         map_func,
>         num_parallel_calls=-1,
>         **kwargs
>     )


:param map_func: an action or a list of actions, for lists None items are skipped


### Class `DataGeneratorMap` {#id}




>     class DataGeneratorMap


Interface for a mapping function for the datagenerator
Use datagenerator.map(object: DataGeneratorMap) to apply.


Attributes
-----=
**```apply_unbatched```**
:      The mapping is performed on batches or not





#### Ancestors (in MRO)

* [abc.ABC](#abc.ABC)



#### Descendants

* [brevettiai.data.data_generator.FileLoader](#brevettiai.data.data_generator.FileLoader)
* [brevettiai.data.image.annotation_pooling.AnnotationPooling](#brevettiai.data.image.annotation_pooling.AnnotationPooling)



#### Class variables



##### Variable `apply_unbatched` {#id}










### Class `FileLoader` {#id}




>     class FileLoader(
>         io=<brevettiai.io.utils.IoTools object>,
>         **data
>     )


Basic File loading module for DataGenerator



#### Ancestors (in MRO)

* [brevettiai.data.data_generator.DataGeneratorMap](#brevettiai.data.data_generator.DataGeneratorMap)
* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.data.image.annotation_loader.AnnotationLoader](#brevettiai.data.image.annotation_loader.AnnotationLoader)
* [brevettiai.data.image.image_loader.ImageLoader](#brevettiai.data.image.image_loader.ImageLoader)



#### Class variables



##### Variable `output_key` {#id}



Type: `str`




##### Variable `path_key` {#id}



Type: `str`




##### Variable `type` {#id}



Type: `typing_extensions.Literal['FileLoader']`





#### Instance variables



##### Variable `apply_unbatched` {#id}




When using in datagenerator, do so on samples, not batches




#### Methods



##### Method `load` {#id}




>     def load(
>         self,
>         path
>     ) ‑> Tuple[Any, Dict[str, Any]]


Loading function, returning data and no metadata about the load


##### Method `load_file_safe` {#id}




>     def load_file_safe(
>         self,
>         path
>     )





### Class `OneHotEncoder` {#id}




>     class OneHotEncoder(
>         classes,
>         input_key='category',
>         output_key='onehot'
>     )


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `encode` {#id}




>     def encode(
>         self,
>         item
>     )





### Class `StratifiedSampler` {#id}




>     class StratifiedSampler(
>         batch_size: int = 32,
>         groupby: list = None,
>         group_weighing: str = 'uniform',
>         max_epoch_samples: int = 1000000000,
>         seed: int = -1
>     )


Base class for serializable modules

<https://en.wikipedia.org/wiki/Stratified_sampling>
:param batch_size: Number of samples per batch
:param groupby: Stratified sample columns to group by when weighing each sample group for sampling
:param group_weighing: Stratfied sampling weighing function to use for weighing the sample groups
supply function or select from ["uniform", "count", "square root", "log"]
:param seed: Seeding of dataset



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `get` {#id}




>     def get(
>         self,
>         samples,
>         shuffle: bool = False,
>         repeat: bool = False,
>         **kwargs
>     ) ‑> brevettiai.data.data_generator.DataGenerator


:param samples: Pandas dataframe with inputs
:param shuffle: Shuffle items in dataset
:param repeat: Repeat samples from dataset
:param max_epoch_samples: Max number of samples per epoch




# Module `brevettiai.data.image` {#id}





## Sub-modules

* [brevettiai.data.image.annotation](#brevettiai.data.image.annotation)
* [brevettiai.data.image.annotation_loader](#brevettiai.data.image.annotation_loader)
* [brevettiai.data.image.annotation_parser](#brevettiai.data.image.annotation_parser)
* [brevettiai.data.image.annotation_pooling](#brevettiai.data.image.annotation_pooling)
* [brevettiai.data.image.bayer_demosaic](#brevettiai.data.image.bayer_demosaic)
* [brevettiai.data.image.feature_calculator](#brevettiai.data.image.feature_calculator)
* [brevettiai.data.image.image_augmenter](#brevettiai.data.image.image_augmenter)
* [brevettiai.data.image.image_loader](#brevettiai.data.image.image_loader)
* [brevettiai.data.image.image_pipeline](#brevettiai.data.image.image_pipeline)
* [brevettiai.data.image.image_processor](#brevettiai.data.image.image_processor)
* [brevettiai.data.image.multi_frame_imager](#brevettiai.data.image.multi_frame_imager)
* [brevettiai.data.image.segmentation_loader](#brevettiai.data.image.segmentation_loader)
* [brevettiai.data.image.utils](#brevettiai.data.image.utils)





## Classes



### Class `ImageKeys` {#id}




>     class ImageKeys








#### Class variables



##### Variable `ANNOTATION` {#id}







##### Variable `BBOX_SIZE_ADJUST` {#id}







##### Variable `BOUNDING_BOX` {#id}







##### Variable `INSIDE_POINTS` {#id}







##### Variable `SIZE` {#id}







##### Variable `ZOOM` {#id}












# Module `brevettiai.data.image.annotation` {#id}










# Module `brevettiai.data.image.annotation_loader` {#id}








## Classes



### Class `AnnotationLoader` {#id}




>     class AnnotationLoader(
>         io=<brevettiai.io.utils.IoTools object>,
>         **data
>     )


Basic File loading module for DataGenerator



#### Ancestors (in MRO)

* [brevettiai.data.data_generator.FileLoader](#brevettiai.data.data_generator.FileLoader)
* [brevettiai.data.data_generator.DataGeneratorMap](#brevettiai.data.data_generator.DataGeneratorMap)
* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `classes` {#id}



Type: `Optional[List[str]]`




##### Variable `mapping` {#id}



Type: `Dict[str, str]`




##### Variable `output_key` {#id}



Type: `str`




##### Variable `path_key` {#id}



Type: `str`




##### Variable `postprocessor` {#id}



Type: `Optional[brevettiai.data.image.image_loader.CropResizeProcessor]`




##### Variable `type` {#id}



Type: `typing_extensions.Literal['AnnotationLoader']`





#### Instance variables



##### Variable `label_space` {#id}











# Module `brevettiai.data.image.annotation_parser` {#id}







## Functions



### Function `bounding_box_area` {#id}




>     def bounding_box_area(
>         bbox
>     )





### Function `draw_contours2` {#id}




>     def draw_contours2(
>         segmentation,
>         label_space,
>         bbox=None,
>         draw_buffer=None,
>         drawContoursArgs=None,
>         **kwargs
>     )


If more than four channels are in the label space only values 1 will be drawn to the segmentation
:param segmentation:
:param label_space:
:param bbox: bbox of annotation to generate [x0,y0,x1,y1]
:param draw_buffer: input draw buffer, use to draw on top of existing images
:param drawContoursArgs: Args for drawContours.. eg thickness to draw non filled contours
:param kwargs: args for make_contours
:return:


### Function `draw_contours2_CHW` {#id}




>     def draw_contours2_CHW(
>         segmentation,
>         label_space,
>         bbox=None,
>         draw_buffer=None,
>         drawContoursArgs=None,
>         **kwargs
>     )


If more than four channels are in the label space only values 1 will be drawn to the segmentation
:param segmentation:
:param label_space:
:param bbox: bbox of annotation to generate [x0,y0,x1,y1]
:param draw_buffer: input draw buffer, use to draw on top of existing images
:param drawContoursArgs: Args for drawContours.. eg thickness to draw non filled contours
:param kwargs: args for make_contours
:return:


### Function `expand_samples_with_annotations` {#id}




>     def expand_samples_with_annotations(
>         samples,
>         verbose=1,
>         key='segmentation_path',
>         how='outer',
>         io=<brevettiai.io.utils.IoTools object>
>     )


Expand samples DataFrame such that each annotation results in a new sample
:param samples: Pandas dataframe with
:param verbose:
:param key: Key in samples with segmentation path
:return:


### Function `get_annotations` {#id}




>     def get_annotations(
>         segmentation_path,
>         io=<brevettiai.io.utils.IoTools object>
>     )





### Function `get_bbox` {#id}




>     def get_bbox(
>         annotation
>     )


Get bounding box bbox from an annotation
:param annotation:
:return:


### Function `get_image_info` {#id}




>     def get_image_info(
>         annotation_path
>     )





### Function `get_points` {#id}




>     def get_points(
>         points,
>         offset=array([0, 0]),
>         scale=1
>     )


Get points from annotation
Offset is given in original coordinates, and is applied before scaling
:param points:
:param offset:
:param scale:
:return:


### Function `make_contour` {#id}




>     def make_contour(
>         points,
>         anno_type,
>         point_size=1,
>         offset=array([0, 0]),
>         scale=1
>     )





### Function `map_segmentations` {#id}




>     def map_segmentations(
>         annotations,
>         segmentation_mapping
>     )





### Function `sample_points_in_annotation` {#id}




>     def sample_points_in_annotation(
>         annotation,
>         tries=100000
>     )





### Function `set_points` {#id}




>     def set_points(
>         points
>     )








# Module `brevettiai.data.image.annotation_pooling` {#id}








## Classes



### Class `AnnotationPooling` {#id}




>     class AnnotationPooling(
>         **data
>     )


Module for pooling annotations to smaller resolution

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.data.image.image_processor.ImageProcessor](#brevettiai.data.image.image_processor.ImageProcessor)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)
* [brevettiai.data.data_generator.DataGeneratorMap](#brevettiai.data.data_generator.DataGeneratorMap)
* [abc.ABC](#abc.ABC)




#### Class variables



##### Variable `input_key` {#id}



Type: `str`




##### Variable `output_key` {#id}



Type: `str`




##### Variable `pool_size` {#id}



Type: `Optional[Tuple[int, int]]`




##### Variable `pooling_algorithms` {#id}



Type: `ClassVar[dict]`




##### Variable `pooling_method` {#id}



Type: `typing_extensions.Literal['max', 'average']`




##### Variable `type` {#id}



Type: `typing_extensions.Literal['AnnotationPooling']`





#### Instance variables



##### Variable `pooling_function` {#id}








#### Static methods



##### `Method validate_pool_size` {#id}




>     def validate_pool_size(
>         v,
>         field
>     )






#### Methods



##### Method `affine_transform` {#id}




>     def affine_transform(
>         self,
>         input_height,
>         input_width
>     )







# Module `brevettiai.data.image.bayer_demosaic` {#id}







## Functions



### Function `bayer_demosaic_layer` {#id}




>     def bayer_demosaic_layer(
>         mode='rgb'
>     )





### Function `get_kernels` {#id}




>     def get_kernels()





### Function `tf_bayer_demosaic` {#id}




>     def tf_bayer_demosaic(
>         x,
>         mode='rgb'
>     )








# Module `brevettiai.data.image.feature_calculator` {#id}








## Classes



### Class `PolygonFeatures` {#id}




>     class PolygonFeatures(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `area` {#id}



Type: `float`




##### Variable `bbox` {#id}



Type: `Tuple[float, float, float, float]`




##### Variable `centroid` {#id}



Type: `Tuple[float, float]`




##### Variable `moments_hu` {#id}



Type: `Tuple[float, float, float, float, float, float, float]`




##### Variable `path_length` {#id}



Type: `float`






#### Static methods



##### `Method calculate_features` {#id}




>     def calculate_features(
>         annotation: PointsAnnotation
>     )








# Module `brevettiai.data.image.image_augmenter` {#id}







## Functions



### Function `affine` {#id}




>     def affine(
>         o1,
>         o2,
>         r1,
>         r2,
>         sc,
>         a,
>         sh1,
>         sh2,
>         t1,
>         t2
>     )


Create N affine transform matrices from output to input (Nx3x3)
for use with tfa.transform
:param o1: Origin 1 (shape / 2 for center)
:param o2: Origin 2
:param r1: Reference scale 1 (1 or -1 for flip)
:param r2: Reference scale 1
:param sc: Augmented scale
:param a: Rotation in radians
:param sh1: Shear 1
:param sh2: Shear 2
:param t1: Translate 1
:param t2: Translate 2
:return: Inverse transformation matrix


### Function `gaussian_blur` {#id}




>     def gaussian_blur(
>         x,
>         sigma
>     )





### Function `get_noise_schema` {#id}




>     def get_noise_schema(
>         ns
>     )





### Function `get_transform_schema` {#id}




>     def get_transform_schema(
>         ns
>     )






## Classes



### Class `ImageAugmentationSchema` {#id}




>     class ImageAugmentationSchema(
>         label='__DEFAULT__',
>         ns='__DEFAULT__',
>         advanced='__DEFAULT__'
>     )






#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.SchemaBuilderFunc](#brevettiai.interfaces.vue_schema_utils.SchemaBuilderFunc)




#### Class variables



##### Variable `advanced` {#id}







##### Variable `label` {#id}







##### Variable `module` {#id}




Base class for serializable modules


##### Variable `ns` {#id}










### Class `ImageAugmenter` {#id}




>     class ImageAugmenter(
>         image_keys=None,
>         label_keys=None,
>         random_transformer: brevettiai.data.image.image_augmenter.RandomTransformer = <brevettiai.data.image.image_augmenter.RandomTransformer object>,
>         image_noise: brevettiai.data.image.image_augmenter.ImageNoise = <brevettiai.data.image.image_augmenter.ImageNoise object>,
>         image_filter: brevettiai.data.image.image_augmenter.ImageFiltering = <brevettiai.data.image.image_augmenter.ImageFiltering object>,
>         image_deformation: brevettiai.data.image.image_augmenter.ImageDeformation = None
>     )


Base class for serializable modules

Image augmentation class, for use with tensorflow and criterion datasets
The call method expects a tensor dict with image and label keys for transformation.
Alternatively, the augmenter may be used by calling transform_images directly

:param image_keys: labels of images to perform augmentation on
:param label_keys: labels of annotations to perform augmentation on
:param random_transformer: A random affine transformation object with RandomTransformer interfaces
:param image_noise: An image noise generation object with ImageNoise interfaces
:param image_filter: An image filter noise generation object with ImageFiltering interfaces
:param image_deformation: A local random image deformation object with ImageDeformation interfaces



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







### Class `ImageDeformation` {#id}




>     class ImageDeformation(
>         alpha: float = 0.0,
>         sigma: float = 0.5,
>         chance: float = 0.5
>     )


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `apply` {#id}




>     def apply(
>         self,
>         x,
>         deformations,
>         probabilities,
>         interpolation='bilinear'
>     )


Elastic deformation of images as described in [Simard2003]_ (with modifications).
.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
     Convolutional Neural Networks applied to Visual Document Analysis", in
     Proc. of the International Conference on Document Analysis and
     Recognition, 2003.

 Based on <https://gist.github.com/erniejunior/601cdf56d2b424757de5>


##### Method `deform_image` {#id}




>     def deform_image(
>         self,
>         inputs,
>         interpolation='bilinear'
>     )





### Class `ImageFiltering` {#id}




>     class ImageFiltering(
>         emboss_strength: tuple = None,
>         avg_blur: tuple = (3, 3),
>         gaussian_blur_sigma: float = 0.5,
>         chance: float = 0.5
>     )


Base class for serializable modules

:param emboss_strength: Embossing in a random direction with a scale chosen in the given range
:param avg_blur: Size of mean filtering kernel
:param gaussian_blur_sigma:
:param chance: The chance of the individual step to be applied



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `conditional_filter` {#id}




>     def conditional_filter(
>         self,
>         inputs
>     )





### Class `ImageNoise` {#id}




>     class ImageNoise(
>         brightness: float = 0.25,
>         contrast: tuple = (0.5, 1.25),
>         hue: float = 0.05,
>         saturation: tuple = (0, 2.0),
>         stddev: float = 0.01,
>         chance: float = 0.5
>     )


Base class for serializable modules

:param brightness: Embossing in a random direction with a scale chosen in the given range
:param contrast: Size of mean filtering kernel
:param hue:
:param saturation:
:param stddev:
:param chance: The chance of the individual step to be applied



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `conditional_noise` {#id}




>     def conditional_noise(
>         self,
>         inputs
>     )





### Class `ImageSaltAndPepper` {#id}




>     class ImageSaltAndPepper(
>         fraction: float = 0.0002,
>         value_range: tuple = None,
>         scale: int = 1,
>         chance: float = 0.5
>     )


Base class for serializable modules

:param brightness: Embossing in a random direction with a scale chosen in the given range
:param contrast: Size of mean filtering kernel
:param hue:
:param saturation:
:param stddev:
:param chance: The chance of the individual step to be applied



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `apply_noise` {#id}




>     def apply_noise(
>         self,
>         inputs
>     )





### Class `RandomTransformer` {#id}




>     class RandomTransformer(
>         chance: float = 0.5,
>         flip_up_down: bool = True,
>         flip_left_right: bool = True,
>         scale: float = 0.2,
>         rotate_chance: float = 0.5,
>         rotate: float = 90,
>         translate_horizontal: float = 0.1,
>         translate_vertical: float = 0.1,
>         shear: float = 0.04,
>         interpolation: str = 'bilinear'
>     )


Base class for serializable modules

Build random transformation matrices for batch of images
:param shape:
:param chance:
:param flip:
:param scale:
:param rotate:
:param translate:
:param shear:
:param interpolation: Resampling interpolation method
:return:



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `set_fill_seg_value` {#id}




>     def set_fill_seg_value(
>         self,
>         fill_value
>     )





##### Method `transform_images` {#id}




>     def transform_images(
>         self,
>         x,
>         A,
>         interpolation=None,
>         fill_seg=False
>     )


:param x: 4D image tensor (batch_size x height x width x channels)
:param A: 3D stack of affine (batch_size x 3 x 3) type is always float32


### Class `ViewGlimpseFromBBox` {#id}




>     class ViewGlimpseFromBBox(
>         bbox_key=None,
>         target_shape: tuple = None,
>         zoom_factor: int = None
>     )


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)





#### Instance variables



##### Variable `bbox_shape` {#id}









### Class `ViewGlimpseFromPoints` {#id}




>     class ViewGlimpseFromPoints(
>         bbox_key=None,
>         target_shape: tuple = None,
>         zoom_factor: int = None,
>         overlap: int = 0.8
>     )


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)





#### Instance variables



##### Variable `bbox_shape` {#id}











# Module `brevettiai.data.image.image_loader` {#id}








## Classes



### Class `CropResizeProcessor` {#id}




>     class CropResizeProcessor(
>         **data: Any
>     )


Baseclass for implementing interface for image proccessors

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.data.image.image_processor.ImageProcessor](#brevettiai.data.image.image_processor.ImageProcessor)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `interpolation` {#id}



Type: `typing_extensions.Literal['bilinear', 'nearest']`




##### Variable `output_height` {#id}



Type: `int`




##### Variable `output_width` {#id}



Type: `int`




##### Variable `roi_height` {#id}



Type: `int`




##### Variable `roi_horizontal_offset` {#id}



Type: `int`




##### Variable `roi_vertical_offset` {#id}



Type: `int`




##### Variable `roi_width` {#id}



Type: `int`




##### Variable `type` {#id}



Type: `typing_extensions.Literal['CropResizeProcessor']`







#### Methods



##### Method `affine_transform` {#id}




>     def affine_transform(
>         self,
>         input_height,
>         input_width
>     )





##### Method `bbox` {#id}




>     def bbox(
>         self,
>         input_height,
>         input_width
>     )


Calculate bounding box specified in pixel coordinates [y1, x1, y2, x2]
The points both being included in the region of interest


##### Method `crop_size` {#id}




>     def crop_size(
>         self,
>         input_height,
>         input_width
>     )





##### Method `output_size` {#id}




>     def output_size(
>         self,
>         input_height,
>         input_width
>     )


Calculated output size of output after postprocessing, given input image sizes


### Class `ImageLoader` {#id}




>     class ImageLoader(
>         io=<brevettiai.io.utils.IoTools object>,
>         **data
>     )


Basic File loading module for DataGenerator



#### Ancestors (in MRO)

* [brevettiai.data.data_generator.FileLoader](#brevettiai.data.data_generator.FileLoader)
* [brevettiai.data.data_generator.DataGeneratorMap](#brevettiai.data.data_generator.DataGeneratorMap)
* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `channels` {#id}



Type: `typing_extensions.Literal[0, 1, 3, 4]`




##### Variable `output_key` {#id}



Type: `str`




##### Variable `postprocessor` {#id}



Type: `Optional[brevettiai.data.image.image_loader.CropResizeProcessor]`




##### Variable `type` {#id}



Type: `typing_extensions.Literal['ImageLoader']`







#### Methods



##### Method `output_shape` {#id}




>     def output_shape(
>         self,
>         image_height=None,
>         image_width=None
>     )





### Class `ScalingProcessor` {#id}




>     class ScalingProcessor(
>         **data: Any
>     )


Baseclass for implementing interface for image proccessors

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.data.image.image_processor.ImageProcessor](#brevettiai.data.image.image_processor.ImageProcessor)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `type` {#id}



Type: `typing_extensions.Literal['ScalingProcessor']`









# Module `brevettiai.data.image.image_pipeline` {#id}








## Classes



### Class `ImagePipeline` {#id}




>     class ImagePipeline(
>         target_size: tuple = None,
>         rois: tuple = None,
>         roi_mode: str = 'concatenate_height',
>         path_key: str = 'path',
>         output_key: str = 'img',
>         color_mode: str = 'rgb',
>         segmentation: brevettiai.data.image.segmentation_loader.SegmentationLoader = None,
>         keep_aspect_ratio=False,
>         rescale: str = None,
>         resize_method: str = 'bilinear',
>         antialias: bool = False,
>         padding_mode: str = 'CONSTANT',
>         center_padding: bool = False,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Base class for serializable modules

:param target_size: target size of images
:param rois: Region of interest(s) (((x11,y11),(x12,y12)),..., ((xn1,yn1),(xn2,yn2)))
:param roi_mode: Treatment of rois (None, ROI_MODE_CONCAT_HEIGHT, ROI_MODE_TIMESERIES)
:param path_key:
:param output_key:
:param color_mode: Color mode of images (greyscale, bayer, rgb)
:param segmentation: SegmentationLoader arguments or object
:param keep_aspect_ratio: keep the aspect ratio during resizing of image
:param rescale: rescaling mode (None [0,255], imagenet [-1,1], unit [0,1])
:param resize_method: resizing method
:param antialias: Apply antialiasing when scaling
:param padding_mode: Padding mode (CONSTANT, REFLECT, SYMMETRIC) applied with tf.pad
:param center_padding: Determine if padding should be centered
:param io:
:param kwargs:



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)




#### Class variables



##### Variable `ROI_MODE` {#id}







##### Variable `ROI_MODE_CONCAT_HEIGHT` {#id}







##### Variable `ROI_MODE_TIMESERIES` {#id}








#### Instance variables



##### Variable `output_shape` {#id}







##### Variable `segmentation` {#id}








#### Static methods



##### `Method from_config` {#id}




>     def from_config(
>         config
>     )





##### `Method get_output_spec` {#id}




>     def get_output_spec(
>         rois,
>         roi_mode,
>         dtype=tf.float32
>     )






#### Methods



##### Method `get_rescaling` {#id}




>     def get_rescaling(
>         self
>     )


returns scale, offset


##### Method `load_images` {#id}




>     def load_images(
>         self,
>         paths,
>         metadata
>     )


Load batch of images given tensor of paths


##### Method `to_image_loader` {#id}




>     def to_image_loader(
>         self
>     )


Build ImageLoader and SegmentationLoader from ImagePipeline




# Module `brevettiai.data.image.image_processor` {#id}








## Classes



### Class `ImageProcessor` {#id}




>     class ImageProcessor(
>         **data: Any
>     )


Baseclass for implementing interface for image proccessors

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.data.image.annotation_pooling.AnnotationPooling](#brevettiai.data.image.annotation_pooling.AnnotationPooling)
* [brevettiai.data.image.image_loader.CropResizeProcessor](#brevettiai.data.image.image_loader.CropResizeProcessor)
* [brevettiai.data.image.image_loader.ScalingProcessor](#brevettiai.data.image.image_loader.ScalingProcessor)



#### Class variables



##### Variable `type` {#id}



Type: `str`






#### Static methods



##### `Method affine_transform` {#id}




>     def affine_transform(
>         input_height,
>         input_width
>     )






#### Methods



##### Method `process` {#id}




>     def process(
>         self,
>         image
>     )


Process image according to processor




# Module `brevettiai.data.image.multi_frame_imager` {#id}








## Classes



### Class `MultiFrameImager` {#id}




>     class MultiFrameImager(
>         **kwargs
>     )


Module to concatenate multiple image frames to a sequence
call generate_paths and set_image_pipeline before use

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `frames` {#id}



Type: `List[int]`





#### Instance variables



##### Variable `apply_unbatched` {#id}







##### Variable `output_shape` {#id}









#### Methods



##### Method `frame_index_format` {#id}




>     def frame_index_format(
>         self,
>         is_prefixed
>     )


Return format of frame index


##### Method `generate_paths` {#id}




>     def generate_paths(
>         self,
>         df
>     )





##### Method `loading_extra_frames` {#id}




>     def loading_extra_frames(
>         self
>     )





##### Method `set_image_pipeline` {#id}




>     def set_image_pipeline(
>         self,
>         image_pipeline
>     )







# Module `brevettiai.data.image.segmentation_loader` {#id}








## Classes



### Class `SegmentationLoader` {#id}




>     class SegmentationLoader(
>         classes: list,
>         mapping: dict = None,
>         image_pipeline=None,
>         sparse=False,
>         input_key='segmentation_path',
>         output_key='segmentation'
>     )


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `build_label_space` {#id}




>     def build_label_space(
>         self,
>         sparse=None
>     )





##### Method `load_segmentations` {#id}




>     def load_segmentations(
>         self,
>         paths,
>         input_image_shape,
>         metadata
>     )





##### Method `set_image_pipeline` {#id}




>     def set_image_pipeline(
>         self,
>         image_pipeline
>     )







# Module `brevettiai.data.image.utils` {#id}







## Functions



### Function `alpha_blend` {#id}




>     def alpha_blend(
>         x,
>         y,
>         alpha=0.4
>     )





### Function `color_mode_transformation` {#id}




>     def color_mode_transformation(
>         img,
>         color_mode
>     )


Transform images by color_mode


### Function `get_tile_func` {#id}




>     def get_tile_func(
>         path,
>         tile_format,
>         tile_size,
>         tile_overlap,
>         channels,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Utility to enclose tile getter func in load_image
:param path:
:param tile_format:
:param tile_size:
:param channels:
:param io:
:return:


### Function `image_view_transform` {#id}




>     def image_view_transform(
>         x,
>         target_size,
>         scale=1,
>         offset=0,
>         **kwargs
>     )


Transform image such that is has the target size and is scaled accordingly
:param x: Image tensor
:param target_size: Target size
:param scale:
:param offset:
:param kwargs: kwargs for resize_image_with_crop_and_pad
:return:


### Function `load_bcimg_frame` {#id}




>     def load_bcimg_frame(
>         path,
>         tile_format,
>         x,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Load bcimage sequence 'tile'


### Function `load_bcimg_json` {#id}




>     def load_bcimg_json(
>         path,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Load bcimage json header for sequential data without prefixed frame names


### Function `load_bcimg_json_prefixed` {#id}




>     def load_bcimg_json_prefixed(
>         path,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Load bcimage json header for sequential data with prefixed frame names


### Function `load_dzi` {#id}




>     def load_dzi(
>         path,
>         zoom=-1,
>         io=<brevettiai.io.utils.IoTools object>
>     )





### Function `load_dzi_json` {#id}




>     def load_dzi_json(
>         path,
>         zoom=-1,
>         io=<brevettiai.io.utils.IoTools object>
>     )





### Function `load_dzi_tile` {#id}




>     def load_dzi_tile(
>         path,
>         tile_format,
>         x,
>         y,
>         io=<brevettiai.io.utils.IoTools object>
>     )





### Function `load_image` {#id}




>     def load_image(
>         path: str,
>         metadata: dict,
>         channels: int,
>         color_mode: str,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Build tf function to load image from a path
:param channels:
:param color_mode:
:param io:
:return: function returning image after color mode transformations


### Function `load_segmentation` {#id}




>     def load_segmentation(
>         path: str,
>         metadata: dict,
>         shape,
>         label_space,
>         io=<brevettiai.io.utils.IoTools object>
>     )


:param path:
:param metadata:
:param shape: shape of
:param label_space:
:param io:
:return:


### Function `pad_image_to_size` {#id}




>     def pad_image_to_size(
>         x,
>         target_size,
>         center_padding=False,
>         **kwargs
>     )


Pad an image to a target size, **kwargs are used to tf.pad
:param x:
:param target_size:
:param kwargs:
:return:


### Function `rescale` {#id}




>     def rescale(
>         x,
>         scale,
>         offset,
>         dtype=tf.float32
>     )


Rescale tensor
:param x:
:param scale:
:param offset:
:param dtype:
:return:


### Function `resize_image_with_crop_and_pad` {#id}




>     def resize_image_with_crop_and_pad(
>         x,
>         target_size,
>         resize_method,
>         keep_aspect_ratio,
>         antialias,
>         padding_mode,
>         center_padding=False
>     )


Resize image with cropping and padding
:param x:
:param target_size:
:param resize_method:
:param keep_aspect_ratio:
:param antialias:
:param padding_mode:
:param center_padding:
:return:


### Function `roi_selection` {#id}




>     def roi_selection(
>         x,
>         rois=None,
>         crops_joiner=None
>     )


Create image crops dependent on BOUNDING_BOX specification
:param x: Image tensor
:param rois:
:param crops_joiner: Crops joining function; could be tf.stack, tf.concat, etc.
:return: list of image crops


### Function `tile2d` {#id}




>     def tile2d(
>         x,
>         grid=(10, 10)
>     )


Function to tile numpy array to plane, eg. images along the first axis
:param x: numpy array to tile
:param grid: size of the grid in tiles
:return: tiled





# Module `brevettiai.data.sample_integrity` {#id}







## Functions



### Function `load_sample_identification` {#id}




>     def load_sample_identification(
>         df,
>         path,
>         column='purpose',
>         io=<brevettiai.io.utils.IoTools object>,
>         **kwargs
>     )


Load and join sample identification information onto dataframe of samples
:param df: sample dataframe
:param path: path to sample id file
:param column: name of split column
:param kwargs: extra args for io_tools.read_file
:return: df, extra_ids


### Function `merge_sample_identification` {#id}




>     def merge_sample_identification(
>         df,
>         dfid,
>         on='etag'
>     )


Merge sample identification traits onto dataframe, such that values (excluding NA) are transfered to the dataframe
:param df: Dataframe
:param dfid: identification dataframe, with index as parameter named by on, by default 'etag'
:param on: column name on df to match with identification
:return: df, extra_ids (merged dataframe and ids, and ids not present among samples


### Function `save_sample_identification` {#id}




>     def save_sample_identification(
>         df,
>         path,
>         known_ids=None,
>         column='purpose',
>         io=<brevettiai.io.utils.IoTools object>
>     )






## Classes



### Class `SampleSplit` {#id}




>     class SampleSplit(
>         stratification: list = None,
>         uniqueness: list = None,
>         split: float = 0.8,
>         seed: int = -1,
>         mode='sorted_permutation'
>     )


Base class for serializable modules

:param stratification: As regex string performed on df.path or list selecting columns
:param uniqueness: As regex string performed on df.path or list selecting columns
:param split: fraction of samples to apply the purpose on
:param seed: seeding for assignment
:param mode: ' or 'murmurhash3'
:return:



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)




#### Class variables



##### Variable `MODE_MURMURHASH3` {#id}







##### Variable `MODE_SORTED_PERMUTATION` {#id}










#### Methods



##### Method `assign` {#id}




>     def assign(
>         self,
>         df,
>         purpose='train',
>         remainder=None,
>         column='purpose'
>     )


Assign purpose column randomly to non-assigned samples based on stratification, uniqueness and split strategy.

Definitions:
* Stratification: Grouping of samples which should be treated as individual groups.
meaning every group must be split according to the sample split target percentage,
and uniqueness is performed on a per group basis
* Uniqueness: grouping of samples which must be treated as a single sample, thus be assigned the same purpose.

:param df: pd.DataFrame of samples if purpose column does not exist it is added
:param purpose: purpose to be assigned
:param remainder: purpose to assign remainder samples, or None to leave unassigned
:param column: column for assignment of split category


##### Method `update_unassigned` {#id}




>     def update_unassigned(
>         self,
>         df,
>         id_path,
>         purpose='train',
>         remainder='devel',
>         column='purpose',
>         io=<brevettiai.io.utils.IoTools object>
>     )


Updates sample purpose in id_path that may hold previous dataset splits and sample ids
Unassigned samples are also assigned and id_path is updated
:param df: pd.DataFrame containing the samples
:param id_path: path to the identification csv file
:param purpose: Purpose to assign
:param remainder: Purpose to assign to remainder or none to leave unassigned
:param column: Column to assign split purposes to
:return:




# Module `brevettiai.data.sample_tools` {#id}







## Functions



### Function `dataset_meta` {#id}




>     def dataset_meta(
>         datasets,
>         tags
>     )


Build dataset meta dataframe from datasets and tags tree
:param datasets:
:param tags:
:return:


### Function `get_grid_bboxes` {#id}




>     def get_grid_bboxes(
>         bbox,
>         size,
>         tile_size=(1024, 1024),
>         overlap=128,
>         num_tile_steps: int = 1,
>         max_steps: int = -1
>     )


Get tiled bounding boxes with overlaps, the last row/column will have a larger overlap to fit the image
:param bbox:
:param size:
:param tile_size:
:param overlap:
:param num_tile_steps:
:param max_steps:
:return:


### Function `get_samples` {#id}




>     def get_samples(
>         datasets,
>         target,
>         *args,
>         **kwargs
>     )


Utility function for getting samples across multiple datasets by sample files
:param datasets:
:param target:
:param args:
:param kwargs:
:return:


### Function `join_dataset_meta` {#id}




>     def join_dataset_meta(
>         df,
>         datasets,
>         tags
>     )


join dataset meta
:param df: sample dataframe with dataset_id to join on
:param datasets: Dataset objects with metadata
:param tags: tag root tree, to find parent tags
:return: df, name/column_id dictionary


### Function `save_samples` {#id}




>     def save_samples(
>         datasets,
>         target,
>         df
>     )






## Classes



### Class `BrevettiDatasetSamples` {#id}




>     class BrevettiDatasetSamples(
>         classes: list = None,
>         class_mapping: dict = None,
>         annotations: dict = None,
>         calculate_md5: bool = False,
>         walk: bool = True,
>         samples_file_name: str = None,
>         contains_column: str = None,
>         contains_regex: str = None
>     )


Base class for serializable modules

:param class_mapping: dict of mapping from path to (category) class. See example for description
:param classes: Force samples to be of the categories in this list



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `get_image_samples` {#id}




>     def get_image_samples(
>         self,
>         datasets,
>         *args,
>         **kwargs
>     )


:param sample_filter: Filter samples by regex
:param annotations: boolean, or dict Load annotation paths
:param kwargs:
:return: Dataframe of samples

Mapping from path to category:
Start from the leaf folder, and work towards the dataset root folder. If folder in class_mapping then apply
its key as the category. If no match is found, apply leaf folder name
Example:
class_mapping={
"A": ["Tilted"],
"B": ["Tilted"],
"1": ["Crimp"]
}
If classes is True or a list/set of categories. The filter is applied after the mapping.


##### Method `get_samples` {#id}




>     def get_samples(
>         self,
>         datasets,
>         walk=None,
>         *args,
>         **kwargs
>     )


Utility function for getting samples across multiple datasets by sample files
:param datasets:
:param target:
:param args:
:param kwargs:
:return:




# Module `brevettiai.data.tf_utils` {#id}







## Functions



### Function `dataset_from_pandas` {#id}




>     def dataset_from_pandas(
>         df
>     )


Build a tensorflow generator dataset from a pandas dataframe allowing tuples of different sizes in each sample
:param df:
:return:


### Function `fill_empty` {#id}




>     def fill_empty(
>         df
>     )





### Function `tf_dataset_metadata` {#id}




>     def tf_dataset_metadata(
>         df
>     )


Generate tf dataset metadata-object from pandas dataframe
:param df:
:return:


### Function `unpack` {#id}




>     def unpack(
>         obj
>     )






## Classes



### Class `NumpyStringIterator` {#id}




>     class NumpyStringIterator(
>         dataset
>     )


Iterator over a dataset with elements converted to numpy. and strings decoded







#### Static methods



##### `Method parser` {#id}




>     def parser(
>         x
>     )






### Class `TfEncoder` {#id}




>     class TfEncoder(
>         *,
>         skipkeys=False,
>         ensure_ascii=True,
>         check_circular=True,
>         allow_nan=True,
>         sort_keys=False,
>         indent=None,
>         separators=None,
>         default=None
>     )


Extensible JSON <http://json.org> encoder for Python data structures.

Supports the following objects and types by default:

+-------------------+---------------+
| Python            | JSON          |
+===================+===============+
| dict              | object        |
+-------------------+---------------+
| list, tuple       | array         |
+-------------------+---------------+
| str               | string        |
+-------------------+---------------+
| int, float        | number        |
+-------------------+---------------+
| True              | true          |
+-------------------+---------------+
| False             | false         |
+-------------------+---------------+
| None              | null          |
+-------------------+---------------+

To extend this to recognize other objects, subclass and implement a
<code>.default()</code> method with another method that returns a serializable
object for <code>o</code> if possible, otherwise it should call the superclass
implementation (to raise <code>TypeError</code>).

Constructor for JSONEncoder, with sensible defaults.

If skipkeys is false, then it is a TypeError to attempt
encoding of keys that are not str, int, float or None.  If
skipkeys is True, such items are simply skipped.

If ensure_ascii is true, the output is guaranteed to be str
objects with all incoming non-ASCII characters escaped.  If
ensure_ascii is false, the output can contain non-ASCII characters.

If check_circular is true, then lists, dicts, and custom encoded
objects will be checked for circular references during encoding to
prevent an infinite recursion (which would cause an OverflowError).
Otherwise, no such check takes place.

If allow_nan is true, then NaN, Infinity, and -Infinity will be
encoded as such.  This behavior is not JSON specification compliant,
but is consistent with most JavaScript based encoders and decoders.
Otherwise, it will be a ValueError to encode such floats.

If sort_keys is true, then the output of dictionaries will be
sorted by key; this is useful for regression tests to ensure
that JSON serializations can be compared on a day-to-day basis.

If indent is a non-negative integer, then JSON array
elements and object members will be pretty-printed with that
indent level.  An indent level of 0 will only insert newlines.
None is the most compact representation.

If specified, separators should be an (item_separator, key_separator)
tuple.  The default is (', ', ': ') if *indent* is <code>None</code> and
(',', ': ') otherwise.  To get the most compact JSON representation,
you should specify (',', ':') to eliminate whitespace.

If specified, default is a function that gets called for objects
that can't otherwise be serialized.  It should return a JSON encodable
version of the object or raise a <code>TypeError</code>.



#### Ancestors (in MRO)

* [json.encoder.JSONEncoder](#json.encoder.JSONEncoder)







#### Methods



##### Method `default` {#id}




>     def default(
>         self,
>         obj
>     )


Implement this method in a subclass such that it returns
a serializable object for <code>o</code>, or calls the base implementation
(to raise a <code>TypeError</code>).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)




# Module `brevettiai.interfaces` {#id}





## Sub-modules

* [brevettiai.interfaces.aws](#brevettiai.interfaces.aws)
* [brevettiai.interfaces.facets_atlas](#brevettiai.interfaces.facets_atlas)
* [brevettiai.interfaces.pivot](#brevettiai.interfaces.pivot)
* [brevettiai.interfaces.raygun](#brevettiai.interfaces.raygun)
* [brevettiai.interfaces.remote_monitor](#brevettiai.interfaces.remote_monitor)
* [brevettiai.interfaces.sagemaker](#brevettiai.interfaces.sagemaker)
* [brevettiai.interfaces.vegalite_charts](#brevettiai.interfaces.vegalite_charts)
* [brevettiai.interfaces.vue_schema_utils](#brevettiai.interfaces.vue_schema_utils)







# Module `brevettiai.interfaces.aws` {#id}







## Functions



### Function `parse_sts_assume_role_response` {#id}




>     def parse_sts_assume_role_response(
>         response,
>         platform
>     )






## Classes



### Class `AWSConfigCredentials` {#id}




>     class AWSConfigCredentials(
>         aws_config_path: str = 'C:\\Users\\emtyg\\.aws',
>         endpoint: str = None
>     )


AWSConfigCredentials(aws_config_path: str = 'C:\\Users\\emtyg\\.aws', endpoint: str = None)



#### Ancestors (in MRO)

* [brevettiai.io.credentials.Credentials](#brevettiai.io.credentials.Credentials)
* [abc.ABC](#abc.ABC)




#### Class variables



##### Variable `aws_config_path` {#id}



Type: `str`




##### Variable `endpoint` {#id}



Type: `str`







#### Methods



##### Method `get_aws_credentials_from_config_file` {#id}




>     def get_aws_credentials_from_config_file(
>         self
>     )





##### Method `get_credentials` {#id}




>     def get_credentials(
>         self,
>         resource_id,
>         resource_type='dataset',
>         mode='r'
>     )





##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         type,
>         user,
>         secret,
>         **kwargs
>     )







# Module `brevettiai.interfaces.facets_atlas` {#id}







## Functions



### Function `build_facets` {#id}




>     def build_facets(
>         dataset,
>         facet_dive,
>         facet_sprite=None,
>         count=4096,
>         exclude_rows=None
>     )


Build facets files
:param dataset:
:param facet_dive: path to facets dive json file or facets dive folder path
:param facet_sprite: path to facets image sprite path
:param count: max count of items
:return:


### Function `create_atlas` {#id}




>     def create_atlas(
>         dataset,
>         count=None
>     )








# Module `brevettiai.interfaces.pivot` {#id}







## Functions



### Function `export_pivot_table` {#id}




>     def export_pivot_table(
>         pivot_dir,
>         df,
>         fields=None,
>         datasets=None,
>         tags=None,
>         rows=None,
>         cols=None,
>         **data_args
>     )


Build and export pivot table using :py:func:pivot_data and :py:func:pivot_fields methods
:param pivot_dir:
:param df:
:param fields:
:param datasets:
:param tags:
:param rows:
:param cols:
:return:


### Function `get_default_fields` {#id}




>     def get_default_fields(
>         df
>     )


Build default pivot fields structure from dataframe
:param df:
:return:


### Function `pivot_data` {#id}




>     def pivot_data(
>         df,
>         fields,
>         datasets=None,
>         tags=None,
>         agg=None
>     )


Build pivot ready dataframe with precalculated object groups
:param df: sample dataframe with dataset_id to join on if datasets and tags are not None
:param fields: field dict, {key:label,...} updated with metadata fields if datasets and tags are not None
:param datasets: datasets to build metadata from
:param tags: tag root tree, to find parent tags
:param agg: Aggregate parameter dictionary, uses count column as default (weight 1 for all samples if nonexistent)
:return: vue-pivot-table export ready dataframe


### Function `pivot_fields` {#id}




>     def pivot_fields(
>         fields,
>         rows=None,
>         cols=None
>     )


Build pivot export fields dict
:param fields: field dict, {key:label,...}
:param rows: iterable of field keys to start in row selector
:param cols: iterable of field keys to start in column selector
:return: vue-pivot-table fields dict





# Module `brevettiai.interfaces.raygun` {#id}







## Functions



### Function `object_extractor` {#id}




>     def object_extractor(
>         types,
>         exc_tb=None,
>         prep_func=None
>     )





### Function `prep_criterion_config` {#id}




>     def prep_criterion_config(
>         config
>     )





### Function `setup_raygun` {#id}




>     def setup_raygun(
>         api_key=None,
>         force=True
>     )








# Module `brevettiai.interfaces.remote_monitor` {#id}








## Classes



### Class `RemoteMonitor` {#id}




>     class RemoteMonitor(
>         root='http://localhost:9000',
>         path='/publish/epoch/end/',
>         field='data',
>         headers=None,
>         send_as_json=False
>     )


Callback used to stream events to a server.

Requires the <code>requests</code> library.
Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
HTTP POST, with a <code>data</code> argument which is a
JSON-encoded dictionary of event data.
If send_as_json is set to True, the content type of the request will be
application/json. Otherwise the serialized JSON will be sent within a form.


Arguments
-----=
root: String; root url of the target server.
path: String; path relative to <code>root</code> to which the events will be sent.
field: String; JSON field under which the data will be stored.
    The field is used only if the payload is sent within a form
    (i.e. send_as_json is set to False).
headers: Dictionary; optional custom HTTP headers.
send_as_json: Boolean; whether the request should be
    sent as application/json.



#### Ancestors (in MRO)

* [keras.callbacks.Callback](#keras.callbacks.Callback)







#### Methods



##### Method `on_epoch_end` {#id}




>     def on_epoch_end(
>         self,
>         epoch,
>         logs=None
>     )


Called at the end of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


Args
-----=
**```epoch```**
:   Integer, index of epoch.


**```logs```**
:   Dict, metric results for this training epoch, and for the
    validation epoch if validation is performed. Validation result keys
    are prefixed with <code>val\_</code>. For training epoch, the values of the


 <code>Model</code>'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
   0.7}`.




# Module `brevettiai.interfaces.sagemaker` {#id}







## Functions



### Function `fetch_aws_credentials` {#id}




>     def fetch_aws_credentials()





### Function `load_hyperparameters_cmd_args` {#id}




>     def load_hyperparameters_cmd_args(
>         hyperparameter_path='/opt/ml/input/config/hyperparameters.json'
>     )






## Classes



### Class `SagemakerCredentials` {#id}




>     class SagemakerCredentials


Abstract class for credential managers



#### Ancestors (in MRO)

* [brevettiai.io.credentials.Credentials](#brevettiai.io.credentials.Credentials)
* [abc.ABC](#abc.ABC)







#### Methods



##### Method `get_credentials` {#id}




>     def get_credentials(
>         self,
>         resource_id,
>         resource_type='dataset',
>         mode='r'
>     )





##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         type,
>         user,
>         secret,
>         **kwargs
>     )







# Module `brevettiai.interfaces.vegalite_charts` {#id}







## Functions



### Function `dataset_summary` {#id}




>     def dataset_summary(
>         samples
>     )





### Function `make_security_selection` {#id}




>     def make_security_selection(
>         devel_pred_output,
>         classes
>     )





### Function `make_selector_chart` {#id}




>     def make_selector_chart(
>         df,
>         x_name,
>         y_name,
>         chart_text,
>         selector,
>         color='red',
>         size=10,
>         scale_type='linear'
>     )








# Module `brevettiai.interfaces.vue_schema_utils` {#id}







## Functions



### Function `apply_schema_to_model` {#id}




>     def apply_schema_to_model(
>         schema,
>         model=None,
>         check_required_fields=True
>     )





### Function `build_from_pydantic_schema` {#id}




>     def build_from_pydantic_schema(
>         schema,
>         definitions,
>         namespace=None
>     )





### Function `checkbox` {#id}




>     def checkbox(
>         label,
>         model,
>         default,
>         required=False,
>         **kwargs
>     )





### Function `checklist` {#id}




>     def checklist(
>         label,
>         model,
>         default,
>         values,
>         required=False,
>         dropdown=True,
>         **kwargs
>     )





### Function `field_class_mapping` {#id}




>     def field_class_mapping(
>         label='Class mapping',
>         model='class_mapping',
>         default='',
>         required=False,
>         hint='Json mapping from folder name to class',
>         **kwargs
>     )





### Function `field_classes` {#id}




>     def field_classes(
>         label='Classes as json list',
>         model='classes',
>         default='',
>         required=False,
>         **kwargs
>     )





### Function `from_pydantic_model` {#id}




>     def from_pydantic_model(
>         model
>     )


Build vue schema from pydantic model


### Function `generate_application_schema` {#id}




>     def generate_application_schema(
>         schema,
>         path='model/settings-schema.json',
>         manifest_path=None
>     )


Generate application schema and manifest files from schema dictionary or SchemaBuilder object
:param schema: schema dictionary or SchemaBuilder object
:param path: target path for schema
:param manifest_path: set to "MANIFEST.in" to export manifest
:return:


### Function `label` {#id}




>     def label(
>         label,
>         **kwargs
>     )





### Function `number_input` {#id}




>     def number_input(
>         label,
>         model,
>         default,
>         required=False,
>         min=0,
>         max=100,
>         step=1,
>         **kwargs
>     )





### Function `parse_json` {#id}




>     def parse_json(
>         x
>     )





### Function `parse_settings_args` {#id}




>     def parse_settings_args(
>         schema,
>         args=None
>     )





### Function `select` {#id}




>     def select(
>         label,
>         model,
>         default,
>         values,
>         required=False,
>         json=False,
>         **kwargs
>     )





### Function `str2bool` {#id}




>     def str2bool(
>         v
>     )





### Function `text_area` {#id}




>     def text_area(
>         label,
>         model,
>         default,
>         required=False,
>         hint='',
>         max=5000,
>         placeholder='',
>         rows=4,
>         json=False,
>         **kwargs
>     )





### Function `text_input` {#id}




>     def text_input(
>         label,
>         model,
>         default='',
>         required=False,
>         json=False,
>         **kwargs
>     )





### Function `update_schema` {#id}




>     def update_schema(
>         schema,
>         settings,
>         ignore=None,
>         field_values=None
>     )





### Function `vue_dtype` {#id}




>     def vue_dtype(
>         field
>     )






## Classes



### Class `SchemaBuilder` {#id}




>     class SchemaBuilder(
>         fields=None,
>         presets=None,
>         modules=None,
>         advanced=True,
>         namespace=None
>     )


Helper object for generating schemas






#### Instance variables



##### Variable `schema` {#id}








#### Static methods



##### `Method from_schema` {#id}




>     def from_schema(
>         schema
>     )






#### Methods



##### Method `add_field` {#id}




>     def add_field(
>         self,
>         field,
>         **kwargs
>     )





##### Method `add_preset` {#id}




>     def add_preset(
>         self,
>         preset,
>         field,
>         value
>     )





##### Method `append` {#id}




>     def append(
>         self,
>         item,
>         *args,
>         **kwargs
>     )





##### Method `filter_fields` {#id}




>     def filter_fields(
>         self,
>         incl_fields: list = None,
>         excl_fields: list = None,
>         make_visible: list = None
>     )





##### Method `load_modules` {#id}




>     def load_modules(
>         self,
>         settings
>     )





### Class `SchemaBuilderFunc` {#id}




>     class SchemaBuilderFunc(
>         label='__DEFAULT__',
>         ns='__DEFAULT__',
>         advanced='__DEFAULT__'
>     )







#### Descendants

* [brevettiai.data.image.image_augmenter.ImageAugmentationSchema](#brevettiai.data.image.image_augmenter.ImageAugmentationSchema)



#### Class variables



##### Variable `advanced` {#id}







##### Variable `label` {#id}







##### Variable `module` {#id}







##### Variable `ns` {#id}









#### Static methods



##### `Method schema` {#id}




>     def schema(
>         self,
>         builder,
>         ns,
>         *args,
>         **kwargs
>     )


Overwrite this function to build schema



#### Methods



##### Method `builder` {#id}




>     def builder(
>         self,
>         *args,
>         **kwargs
>     )





### Class `SchemaConfig` {#id}




>     class SchemaConfig(
>         exclude: bool = False
>     )


Configuration object for Vue schema generation





#### Class variables



##### Variable `exclude` {#id}



Type: `bool`







### Class `VueSettingsModule` {#id}




>     class VueSettingsModule


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)



#### Descendants

* [brevettiai.data.data_generator.OneHotEncoder](#brevettiai.data.data_generator.OneHotEncoder)
* [brevettiai.data.data_generator.StratifiedSampler](#brevettiai.data.data_generator.StratifiedSampler)
* [brevettiai.data.image.image_augmenter.ImageAugmenter](#brevettiai.data.image.image_augmenter.ImageAugmenter)
* [brevettiai.data.image.image_augmenter.ImageDeformation](#brevettiai.data.image.image_augmenter.ImageDeformation)
* [brevettiai.data.image.image_augmenter.ImageFiltering](#brevettiai.data.image.image_augmenter.ImageFiltering)
* [brevettiai.data.image.image_augmenter.ImageNoise](#brevettiai.data.image.image_augmenter.ImageNoise)
* [brevettiai.data.image.image_augmenter.ImageSaltAndPepper](#brevettiai.data.image.image_augmenter.ImageSaltAndPepper)
* [brevettiai.data.image.image_augmenter.RandomTransformer](#brevettiai.data.image.image_augmenter.RandomTransformer)
* [brevettiai.data.image.image_augmenter.ViewGlimpseFromBBox](#brevettiai.data.image.image_augmenter.ViewGlimpseFromBBox)
* [brevettiai.data.image.image_augmenter.ViewGlimpseFromPoints](#brevettiai.data.image.image_augmenter.ViewGlimpseFromPoints)
* [brevettiai.data.image.image_pipeline.ImagePipeline](#brevettiai.data.image.image_pipeline.ImagePipeline)
* [brevettiai.data.image.segmentation_loader.SegmentationLoader](#brevettiai.data.image.segmentation_loader.SegmentationLoader)
* [brevettiai.data.sample_integrity.SampleSplit](#brevettiai.data.sample_integrity.SampleSplit)
* [brevettiai.data.sample_tools.BrevettiDatasetSamples](#brevettiai.data.sample_tools.BrevettiDatasetSamples)





#### Static methods



##### `Method from_settings` {#id}




>     def from_settings(
>         settings
>     )





##### `Method get_schema` {#id}




>     def get_schema(
>         namespace=None
>     )


Get vue-form-generator schema


##### `Method to_config` {#id}




>     def to_config(
>         settings
>     )


Parse settings from vue-form-generator json model to python config

Overwrite this if settings data model is different than config data model.
Remember to overwrite to_settings as well to provide the reverse transformation


##### `Method to_schema` {#id}




>     def to_schema(
>         builder: brevettiai.interfaces.vue_schema_utils.SchemaBuilder,
>         name: str,
>         ptype: type,
>         default,
>         **kwargs
>     )


Transform field to vue-form-generator schema fields.

overwrite this to provide custom schemas for your Model


##### `Method to_settings` {#id}




>     def to_settings(
>         config
>     )


Get settings model for vue-schema-generator.

Overwrite this if you have custom field manipulaion in from settings


##### `Method validator` {#id}




>     def validator(
>         x
>     )






#### Methods



##### Method `get_settings` {#id}




>     def get_settings(
>         self
>     )


Get Vue schema settings model for vue-form-generator




# Module `brevettiai.io` {#id}





## Sub-modules

* [brevettiai.io.credentials](#brevettiai.io.credentials)
* [brevettiai.io.h5_metadata](#brevettiai.io.h5_metadata)
* [brevettiai.io.local_io](#brevettiai.io.local_io)
* [brevettiai.io.minio_io](#brevettiai.io.minio_io)
* [brevettiai.io.onnx](#brevettiai.io.onnx)
* [brevettiai.io.openvino](#brevettiai.io.openvino)
* [brevettiai.io.path](#brevettiai.io.path)
* [brevettiai.io.serialization](#brevettiai.io.serialization)
* [brevettiai.io.tf_recorder](#brevettiai.io.tf_recorder)
* [brevettiai.io.utils](#brevettiai.io.utils)







# Module `brevettiai.io.credentials` {#id}








## Classes



### Class `Credentials` {#id}




>     class Credentials


Abstract class for credential managers



#### Ancestors (in MRO)

* [abc.ABC](#abc.ABC)



#### Descendants

* [brevettiai.interfaces.aws.AWSConfigCredentials](#brevettiai.interfaces.aws.AWSConfigCredentials)
* [brevettiai.interfaces.sagemaker.SagemakerCredentials](#brevettiai.interfaces.sagemaker.SagemakerCredentials)
* [brevettiai.io.credentials.CredentialsChain](#brevettiai.io.credentials.CredentialsChain)
* [brevettiai.platform.platform_credentials.JobCredentials](#brevettiai.platform.platform_credentials.JobCredentials)
* [brevettiai.platform.platform_credentials.PlatformDatasetCredentials](#brevettiai.platform.platform_credentials.PlatformDatasetCredentials)






#### Methods



##### Method `get_credentials` {#id}




>     def get_credentials(
>         self,
>         resource_id,
>         resource_type='dataset',
>         mode='r'
>     )





##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         type,
>         user,
>         secret,
>         **kwargs
>     )





### Class `CredentialsChain` {#id}




>     class CredentialsChain(
>         chain: List[brevettiai.io.credentials.Credentials] = <factory>
>     )


Credentials chain grouping a number of credentials into one trying all of them in order



#### Ancestors (in MRO)

* [brevettiai.io.credentials.Credentials](#brevettiai.io.credentials.Credentials)
* [abc.ABC](#abc.ABC)



#### Descendants

* [brevettiai.platform.platform_credentials.DefaultJobCredentialsChain](#brevettiai.platform.platform_credentials.DefaultJobCredentialsChain)



#### Class variables



##### Variable `chain` {#id}



Type: `List[brevettiai.io.credentials.Credentials]`







#### Methods



##### Method `get_credentials` {#id}




>     def get_credentials(
>         self,
>         resource_id,
>         resource_type='dataset',
>         mode='r'
>     )





##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         type,
>         user,
>         secret,
>         **kwargs
>     )





### Class `LoginError` {#id}




>     class LoginError(
>         *args,
>         **kwargs
>     )


Common base class for all non-exit exceptions.



#### Ancestors (in MRO)

* [builtins.Exception](#builtins.Exception)
* [builtins.BaseException](#builtins.BaseException)









# Module `brevettiai.io.h5_metadata` {#id}







## Functions



### Function `get_metadata` {#id}




>     def get_metadata(
>         h5_path
>     )





### Function `save_model` {#id}




>     def save_model(
>         path,
>         model,
>         metadata
>     )





### Function `set_metadata` {#id}




>     def set_metadata(
>         h5_path,
>         metadata
>     )








# Module `brevettiai.io.local_io` {#id}








## Classes



### Class `LocalIO` {#id}




>     class LocalIO










#### Static methods



##### `Method copy` {#id}




>     def copy(
>         src,
>         dst,
>         *args,
>         **kwargs
>     )





##### `Method file_cache` {#id}




>     def file_cache(
>         path,
>         cache_root,
>         data_getter,
>         max_cache_usage_fraction
>     )


Cache file data to local file
:param path: source path
:param cache_root: root of cache location
:param data_getter: function getting the data from th
:param max_cache_usage_fraction: do not fill disk more than this fraction
:return:


##### `Method get_md5` {#id}




>     def get_md5(
>         path
>     )





##### `Method isfile` {#id}




>     def isfile(
>         path
>     )





##### `Method make_dirs` {#id}




>     def make_dirs(
>         path,
>         exist_ok=True
>     )





##### `Method move` {#id}




>     def move(
>         src,
>         dst,
>         *args,
>         **kwargs
>     )





##### `Method read` {#id}




>     def read(
>         path
>     )





##### `Method remove` {#id}




>     def remove(
>         path
>     )





##### `Method walk_visible` {#id}




>     def walk_visible(
>         path
>     )





##### `Method write` {#id}




>     def write(
>         path,
>         content
>     )






#### Methods



##### Method `resolve_access_rights` {#id}




>     def resolve_access_rights(
>         self,
>         *args,
>         **kwargs
>     )





##### Method `walk` {#id}




>     def walk(
>         self,
>         path,
>         exclude_hidden=False,
>         **kwargs
>     )





### Class `safe_open` {#id}




>     class safe_open(
>         path,
>         mode='w+b'
>     )


Temporary file backed storage for safely writing to cache
:param path:
:param mode:










# Module `brevettiai.io.minio_io` {#id}







## Functions



### Function `token_error_fallback` {#id}




>     def token_error_fallback(
>         f,
>         set_client
>     )






## Classes



### Class `MinioIO` {#id}




>     class MinioIO(
>         cache_files: bool = True,
>         credentials=None
>     )








#### Class variables



##### Variable `http_pool` {#id}










#### Methods



##### Method `calculate_md5` {#id}




>     def calculate_md5(
>         self,
>         path
>     )





##### Method `client_factory` {#id}




>     def client_factory(
>         self,
>         prefix,
>         credentials_func
>     )





##### Method `copy` {#id}




>     def copy(
>         self,
>         src,
>         dst,
>         *args,
>         **kwargs
>     )





##### Method `get_client` {#id}




>     def get_client(
>         self,
>         path
>     )





##### Method `get_md5` {#id}




>     def get_md5(
>         self,
>         path
>     )





##### Method `isfile` {#id}




>     def isfile(
>         self,
>         path
>     )





##### Method `make_dirs` {#id}




>     def make_dirs(
>         self,
>         path,
>         exist_ok=True
>     )





##### Method `move` {#id}




>     def move(
>         self,
>         src,
>         dst,
>         *args,
>         **kwargs
>     )





##### Method `read` {#id}




>     def read(
>         self,
>         path,
>         *,
>         client=None
>     )





##### Method `remove` {#id}




>     def remove(
>         self,
>         path
>     )





##### Method `resolve_access_rights` {#id}




>     def resolve_access_rights(
>         self,
>         path,
>         *args,
>         **kwargs
>     )





##### Method `set_route` {#id}




>     def set_route(
>         self,
>         prefix,
>         resource_id,
>         resource_type,
>         mode='r'
>     )





##### Method `stat_object` {#id}




>     def stat_object(
>         self,
>         path
>     )





##### Method `walk` {#id}




>     def walk(
>         self,
>         path,
>         prefix=None,
>         recursive=True,
>         include_object=False,
>         exclude_hidden=False,
>         **kwargs
>     )





##### Method `write` {#id}




>     def write(
>         self,
>         path,
>         content,
>         *,
>         client=None
>     )





### Class `TSPoolManager` {#id}




>     class TSPoolManager(
>         num_pools=10,
>         headers=None,
>         **connection_pool_kw
>     )


Fixed pool manager: <https://github.com/urllib3/urllib3/issues/1252>



#### Ancestors (in MRO)

* [urllib3.poolmanager.PoolManager](#urllib3.poolmanager.PoolManager)
* [urllib3.request.RequestMethods](#urllib3.request.RequestMethods)









# Module `brevettiai.io.onnx` {#id}







## Functions



### Function `export_model` {#id}




>     def export_model(
>         model,
>         output_file=None,
>         inputs_as_nchw: (<class 'list'>, <class 'bool'>) = None,
>         shape_override=None,
>         meta_data:��dict = None
>     )





### Function `input_as_nchw` {#id}




>     def input_as_nchw(
>         function
>     )





### Function `input_output_quantization` {#id}




>     def input_output_quantization(
>         function,
>         dtype=tf.uint8,
>         output_scaling=255
>     )








# Module `brevettiai.io.openvino` {#id}







## Functions



### Function `export_model` {#id}




>     def export_model(
>         model,
>         output_dir=None,
>         shape_override=None,
>         meta_data: dict = None
>     )





### Function `predict` {#id}




>     def predict(
>         tmpdir
>     )








# Module `brevettiai.io.path` {#id}







## Functions



### Function `get_folders` {#id}




>     def get_folders(
>         path,
>         bucket
>     )





### Function `get_sep` {#id}




>     def get_sep(
>         path
>     )





### Function `join` {#id}




>     def join(
>         *paths
>     )


Join os paths and urls
:param paths:
:return:


### Function `movedir` {#id}




>     def movedir(
>         srcdir,
>         targetdir
>     )


Move contents of directory on linux and windows file systems
:param srcdir:
:param targetdir:
:return:


### Function `partition` {#id}




>     def partition(
>         items,
>         func
>     )


Partition a list based on a function
:param items: List of items
:param func: Function to partition after
:return:


### Function `relpath` {#id}




>     def relpath(
>         path,
>         start=None
>     )


Use os.path.relpath for local drives and adapted version for URIs
Might be brittle when using URIs
Return a relative version of a path


### Function `safe_join` {#id}




>     def safe_join(
>         a0,
>         *args
>     )








# Module `brevettiai.io.serialization` {#id}








## Classes



### Class `ObjectJsonEncoder` {#id}




>     class ObjectJsonEncoder(
>         *,
>         skipkeys=False,
>         ensure_ascii=True,
>         check_circular=True,
>         allow_nan=True,
>         sort_keys=False,
>         indent=None,
>         separators=None,
>         default=None
>     )


Extensible JSON <http://json.org> encoder for Python data structures.

Supports the following objects and types by default:

+-------------------+---------------+
| Python            | JSON          |
+===================+===============+
| dict              | object        |
+-------------------+---------------+
| list, tuple       | array         |
+-------------------+---------------+
| str               | string        |
+-------------------+---------------+
| int, float        | number        |
+-------------------+---------------+
| True              | true          |
+-------------------+---------------+
| False             | false         |
+-------------------+---------------+
| None              | null          |
+-------------------+---------------+

To extend this to recognize other objects, subclass and implement a
<code>.default()</code> method with another method that returns a serializable
object for <code>o</code> if possible, otherwise it should call the superclass
implementation (to raise <code>TypeError</code>).

Constructor for JSONEncoder, with sensible defaults.

If skipkeys is false, then it is a TypeError to attempt
encoding of keys that are not str, int, float or None.  If
skipkeys is True, such items are simply skipped.

If ensure_ascii is true, the output is guaranteed to be str
objects with all incoming non-ASCII characters escaped.  If
ensure_ascii is false, the output can contain non-ASCII characters.

If check_circular is true, then lists, dicts, and custom encoded
objects will be checked for circular references during encoding to
prevent an infinite recursion (which would cause an OverflowError).
Otherwise, no such check takes place.

If allow_nan is true, then NaN, Infinity, and -Infinity will be
encoded as such.  This behavior is not JSON specification compliant,
but is consistent with most JavaScript based encoders and decoders.
Otherwise, it will be a ValueError to encode such floats.

If sort_keys is true, then the output of dictionaries will be
sorted by key; this is useful for regression tests to ensure
that JSON serializations can be compared on a day-to-day basis.

If indent is a non-negative integer, then JSON array
elements and object members will be pretty-printed with that
indent level.  An indent level of 0 will only insert newlines.
None is the most compact representation.

If specified, separators should be an (item_separator, key_separator)
tuple.  The default is (', ', ': ') if *indent* is <code>None</code> and
(',', ': ') otherwise.  To get the most compact JSON representation,
you should specify (',', ':') to eliminate whitespace.

If specified, default is a function that gets called for objects
that can't otherwise be serialized.  It should return a JSON encodable
version of the object or raise a <code>TypeError</code>.



#### Ancestors (in MRO)

* [json.encoder.JSONEncoder](#json.encoder.JSONEncoder)







#### Methods



##### Method `default` {#id}




>     def default(
>         self,
>         o
>     )


Implement this method in a subclass such that it returns
a serializable object for <code>o</code>, or calls the base implementation
(to raise a <code>TypeError</code>).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)




# Module `brevettiai.io.tf_recorder` {#id}







## Functions



### Function `generate_dtype_structure` {#id}




>     def generate_dtype_structure(
>         value
>     )





### Function `serialize_composite_structure` {#id}




>     def serialize_composite_structure(
>         value
>     )






## Classes



### Class `TfRecorder` {#id}




>     class TfRecorder(
>         filenames,
>         structure=None,
>         compression_type='GZIP'
>     )


Base class for serializable modules



#### Ancestors (in MRO)

* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)





#### Instance variables



##### Variable `feature_description` {#id}








#### Static methods



##### `Method from_config` {#id}




>     def from_config(
>         config
>     )






#### Methods



##### Method `get_config` {#id}




>     def get_config(
>         self
>     )





##### Method `get_dataset` {#id}




>     def get_dataset(
>         self,
>         *args,
>         **kwargs
>     )





##### Method `parse_dataset` {#id}




>     def parse_dataset(
>         self,
>         x
>     )





##### Method `serialize` {#id}




>     def serialize(
>         self,
>         value
>     )





##### Method `set_structure_from_example` {#id}




>     def set_structure_from_example(
>         self,
>         value
>     )





##### Method `write` {#id}




>     def write(
>         self,
>         value
>     )







# Module `brevettiai.io.utils` {#id}







## Functions



### Function `load_file_safe` {#id}




>     def load_file_safe(
>         x,
>         cache_dir=None,
>         io=<brevettiai.io.utils.IoTools object>
>     )


Load a file safely with and without tensorflow
:param x: path to file
:param cache_dir:
:param io:
:return:



## Classes



### Class `IoTools` {#id}




>     class IoTools(
>         cache_root=None,
>         localio=<brevettiai.io.local_io.LocalIO object>,
>         minio=<brevettiai.io.minio_io.MinioIO object>,
>         max_cache_usage_fraction=0.8,
>         path=<module 'brevettiai.io.path' from 'C:\\Users\\emtyg\\dev\\brevetti\\core\\brevettiai\\io\\path.py'>
>     )


:param cache_path: common cache path
:param localio: path to local file storage backend
:param minio: path to minio backend
:param max_cache_usage_fraction: stop caching when exeeding this usage fraction







#### Static methods



##### `Method factory` {#id}




>     def factory(
>         **kwargs
>     )


Build IoTools with new backends
:param args:
:param kwargs:
:return:


##### `Method get_uri` {#id}




>     def get_uri(
>         path
>     )






#### Methods



##### Method `copy` {#id}




>     def copy(
>         self,
>         src,
>         dst,
>         *args,
>         **kwargs
>     )





##### Method `get_backend` {#id}




>     def get_backend(
>         self,
>         path
>     )





##### Method `get_md5` {#id}




>     def get_md5(
>         self,
>         path
>     )





##### Method `isfile` {#id}




>     def isfile(
>         self,
>         path
>     )





##### Method `make_dirs` {#id}




>     def make_dirs(
>         self,
>         path
>     )





##### Method `move` {#id}




>     def move(
>         self,
>         src,
>         dst,
>         *args,
>         **kwargs
>     )





##### Method `read_file` {#id}




>     def read_file(
>         self,
>         path,
>         cache=None,
>         errors='raise'
>     )





##### Method `remove` {#id}




>     def remove(
>         self,
>         path
>     )





##### Method `resolve_access_rights` {#id}




>     def resolve_access_rights(
>         self,
>         path,
>         *args,
>         **kwargs
>     )





##### Method `set_cache_root` {#id}




>     def set_cache_root(
>         self,
>         root
>     )





##### Method `walk` {#id}




>     def walk(
>         self,
>         path,
>         exclude_hidden=False,
>         **kwargs
>     )





##### Method `write_file` {#id}




>     def write_file(
>         self,
>         path,
>         content
>     )







# Module `brevettiai.model` {#id}





## Sub-modules

* [brevettiai.model.factory](#brevettiai.model.factory)
* [brevettiai.model.losses](#brevettiai.model.losses)
* [brevettiai.model.metadata](#brevettiai.model.metadata)







# Module `brevettiai.model.factory` {#id}





## Sub-modules

* [brevettiai.model.factory.lraspp](#brevettiai.model.factory.lraspp)
* [brevettiai.model.factory.mobilenetv2_backbone](#brevettiai.model.factory.mobilenetv2_backbone)
* [brevettiai.model.factory.segmentation](#brevettiai.model.factory.segmentation)





## Classes



### Class `ModelFactory` {#id}




>     class ModelFactory(
>         **data: Any
>     )


Abstract model factory class

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.model.factory.lraspp.LRASPP2SegmentationHead](#brevettiai.model.factory.lraspp.LRASPP2SegmentationHead)
* [brevettiai.model.factory.mobilenetv2_backbone.MobileNetV2SegmentationBackbone](#brevettiai.model.factory.mobilenetv2_backbone.MobileNetV2SegmentationBackbone)
* [brevettiai.model.factory.segmentation.SegmentationModel](#brevettiai.model.factory.segmentation.SegmentationModel)





#### Static methods



##### `Method custom_objects` {#id}




>     def custom_objects()


Custom objects used by the model



#### Methods



##### Method `build` {#id}




>     def build(
>         self,
>         input_shape,
>         output_shape,
>         **kwargs
>     ) ‑> tensorflow.python.keras.engine.functional.Functional


Function to build segmentation backbone




# Module `brevettiai.model.factory.lraspp` {#id}







## Functions



### Function `ceil_divisible_by_8` {#id}




>     def ceil_divisible_by_8(
>         x
>     )






## Classes



### Class `LRASPP2SegmentationHead` {#id}




>     class LRASPP2SegmentationHead(
>         **data: Any
>     )


Abstract model factory class

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.model.factory.ModelFactory](#brevettiai.model.factory.ModelFactory)
* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `avg_pool_kernel` {#id}



Type: `Tuple[int, int]`




##### Variable `avg_pool_strides` {#id}



Type: `Tuple[int, int]`




##### Variable `bn_momentum` {#id}



Type: `float`




##### Variable `filter_bank_multiplier` {#id}



Type: `int`




##### Variable `output_channels` {#id}



Type: `Optional[int]`




##### Variable `resize_method` {#id}



Type: `str`









# Module `brevettiai.model.factory.mobilenetv2_backbone` {#id}







## Functions



### Function `remap_backbone` {#id}




>     def remap_backbone(
>         bn_momentum,
>         default_regularizer,
>         exchange_padding_on
>     )






## Classes



### Class `MobileNetV2SegmentationBackbone` {#id}




>     class MobileNetV2SegmentationBackbone(
>         **data: Any
>     )


Abstract model factory class

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.model.factory.ModelFactory](#brevettiai.model.factory.ModelFactory)
* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `alpha` {#id}



Type: `float`




##### Variable `bn_momentum` {#id}



Type: `float`




##### Variable `l1_regularization` {#id}



Type: `float`




##### Variable `l2_regularization` {#id}



Type: `float`




##### Variable `output_layers` {#id}



Type: `List[str]`




##### Variable `weights` {#id}



Type: `str`









# Module `brevettiai.model.factory.segmentation` {#id}








## Classes



### Class `SegmentationModel` {#id}




>     class SegmentationModel(
>         **data: Any
>     )


Abstract model factory class

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.model.factory.ModelFactory](#brevettiai.model.factory.ModelFactory)
* [abc.ABC](#abc.ABC)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `backbone_factory` {#id}



Type: `brevettiai.model.factory.ModelFactory`




##### Variable `bn_momentum` {#id}



Type: `float`




##### Variable `classes` {#id}



Type: `List[str]`




##### Variable `head_factory` {#id}



Type: `brevettiai.model.factory.ModelFactory`




##### Variable `resize_method` {#id}



Type: `typing_extensions.Literal['bilinear', 'nearest']`




##### Variable `resize_output` {#id}



Type: `bool`





#### Instance variables



##### Variable `backbone` {#id}







##### Variable `head` {#id}







##### Variable `model` {#id}









#### Methods



##### Method `build` {#id}




>     def build(
>         self,
>         input_shape: Tuple[Optional[int], Optional[int], Optional[int]],
>         **kwargs
>     )


Function to build the segmentation model and return the input and output keras tensors




# Module `brevettiai.model.losses` {#id}







## Functions



### Function `weighted_loss` {#id}




>     def weighted_loss(
>         y_true,
>         y_pred,
>         baseloss,
>         sample_weights,
>         sample_weights_bias,
>         output_weights,
>         **kwargs
>     )






## Classes



### Class `WeightedLossV2` {#id}




>     class WeightedLossV2(
>         baseloss='binary_crossentropy',
>         sample_weights=None,
>         sample_weights_bias=None,
>         output_weights=None,
>         from_logits=False,
>         label_smoothing=0.0,
>         reduction='auto',
>         name='weighted_loss'
>     )


Wraps a loss function in the <code>Loss</code> class.

Initializes <code>LossFunctionWrapper</code> class.


Args
-----=
**```fn```**
:   The loss function to wrap, with signature `fn(y_true, y_pred,
    **kwargs)`.


**```reduction```**
:   Type of <code>tf.keras.losses.Reduction</code> to apply to
    loss. Default value is <code>AUTO</code>. <code>AUTO</code> indicates that the reduction
    option will be determined by the usage context. For almost all cases
    this defaults to <code>SUM\_OVER\_BATCH\_SIZE</code>. When used with
    <code>tf.distribute.Strategy</code>, outside of built-in training loops such as
    <code>tf.keras</code> <code>compile</code> and <code>fit</code>, using <code>AUTO</code> or <code>SUM\_OVER\_BATCH\_SIZE</code>
    will raise an error. Please see this custom training [tutorial](
      <https://www.tensorflow.org/tutorials/distribute/custom_training>) for
        more details.


**```name```**
:   Optional name for the instance.


**```**kwargs```**
:   The keyword arguments that are passed on to <code>fn</code>.





#### Ancestors (in MRO)

* [tensorflow.python.keras.losses.LossFunctionWrapper](#tensorflow.python.keras.losses.LossFunctionWrapper)
* [tensorflow.python.keras.losses.Loss](#tensorflow.python.keras.losses.Loss)









# Module `brevettiai.model.metadata` {#id}





## Sub-modules

* [brevettiai.model.metadata.image_segmentation](#brevettiai.model.metadata.image_segmentation)
* [brevettiai.model.metadata.metadata](#brevettiai.model.metadata.metadata)







# Module `brevettiai.model.metadata.image_segmentation` {#id}








## Classes



### Class `Base64Image` {#id}




>     class Base64Image(
>         ...
>     )


ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using <code>array</code>, <code>zeros</code> or <code>empty</code> (refer
to the See Also section below).  The parameters given here refer to
a low-level method (<code>ndarray(...)</code>) for instantiating an array.

For more information, refer to the <code>numpy</code> module and examine the
methods and attributes of an array.

#### Parameters

(for the __new__ method; see Notes below)

**```shape```** :&ensp;<code>tuple</code> of <code>ints</code>
:   Shape of created array.


**```dtype```** :&ensp;`data-type`, optional
:   Any object that can be interpreted as a numpy data type.


**```buffer```** :&ensp;<code>object exposing buffer interface</code>, optional
:   Used to fill the array with data.


**```offset```** :&ensp;<code>int</code>, optional
:   Offset of array data in buffer.


**```strides```** :&ensp;<code>tuple</code> of <code>ints</code>, optional
:   Strides of data in memory.


**```order```** :&ensp;`{'C', 'F'}`, optional
:   Row-major (C-style) or column-major (Fortran-style) order.

#### Attributes

**```T```** :&ensp;<code>ndarray</code>
:   Transpose of the array.


**```data```** :&ensp;<code>buffer</code>
:   The array's elements, in memory.


**```dtype```** :&ensp;<code>dtype object</code>
:   Describes the format of the elements in the array.


**```flags```** :&ensp;<code>dict</code>
:   Dictionary containing information related to memory use, e.g.,
    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.


**```flat```** :&ensp;<code>numpy.flatiter object</code>
:   Flattened version of the array as an iterator.  The iterator
    allows assignments, e.g., ``x.flat = 3`` (See <code>ndarray.flat</code> for
    assignment examples; TODO).


**```imag```** :&ensp;<code>ndarray</code>
:   Imaginary part of the array.


**```real```** :&ensp;<code>ndarray</code>
:   Real part of the array.


**```size```** :&ensp;<code>int</code>
:   Number of elements in the array.


**```itemsize```** :&ensp;<code>int</code>
:   The memory use of each array element in bytes.


**```nbytes```** :&ensp;<code>int</code>
:   The total number of bytes required to store the array data,
    i.e., ``itemsize * size``.


**```ndim```** :&ensp;<code>int</code>
:   The array's number of dimensions.


**```shape```** :&ensp;<code>tuple</code> of <code>ints</code>
:   Shape of the array.


**```strides```** :&ensp;<code>tuple</code> of <code>ints</code>
:   The step-size required to move from one element to the next in
    memory. For example, a contiguous <code>(3, 4)</code> array of type
    <code>int16</code> in C-order has strides <code>(8, 2)</code>.  This implies that
    to move from element to element in memory requires jumps of 2 bytes.
    To move from row-to-row, one needs to jump 8 bytes at a time
    (``2 * 4``).


**```ctypes```** :&ensp;<code>ctypes object</code>
:   Class containing properties of the array needed for interaction
    with ctypes.


**```base```** :&ensp;<code>ndarray</code>
:   If the array is a view into another array, that array is its <code>base</code>
    (unless that array is also a view).  The <code>base</code> array is where the
    array data is actually stored.

#### See Also

<code>array</code>
:   Construct an array.

<code>zeros</code>
:   Create an array, each element of which is zero.

<code>empty</code>
:   Create an array, but leave its allocated memory unchanged (i.e., it contains "garbage").

<code>dtype</code>
:   Create a data-type.

<code>numpy.typing.NDArray</code>
:   A :term:`generic <generic type>` version of ndarray.

#### Notes

There are two modes of creating an array using <code>\_\_new\_\_</code>:

1. If <code>buffer</code> is None, then only <code>shape</code>, <code>dtype</code>, and <code>order</code>
   are used.
2. If <code>buffer</code> is an object exposing the buffer interface, then
   all keywords are interpreted.

No <code>\_\_init\_\_</code> method is needed because the array is fully initialized
after the <code>\_\_new\_\_</code> method.

#### Examples

These examples illustrate the low-level <code>ndarray</code> constructor.  Refer
to the <code>See Also</code> section above for easier ways of constructing an
ndarray.

First mode, <code>buffer</code> is None:

```python-repl
>>> np.ndarray(shape=(2,2), dtype=float, order='F')
array([[0.0e+000, 0.0e+000], # random
       [     nan, 2.5e-323]])
```


Second mode:

```python-repl
>>> np.ndarray((2,), buffer=np.array([1,2,3]),
...            offset=np.int_().itemsize,
...            dtype=int) # offset = 1*itemsize, i.e. skip first element
array([2, 3])
```




#### Ancestors (in MRO)

* [numpy.ndarray](#numpy.ndarray)






#### Static methods



##### `Method validate_type` {#id}




>     def validate_type(
>         val
>     )






### Class `ImageSegmentationModelMetadata` {#id}




>     class ImageSegmentationModelMetadata(
>         **data: Any
>     )


Metadata for an Image segmentation model

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.model.metadata.metadata.ModelMetadata](#brevettiai.model.metadata.metadata.ModelMetadata)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `Config` {#id}







##### Variable `annotation_loader` {#id}



Type: `brevettiai.data.image.annotation_loader.AnnotationLoader`




##### Variable `annotation_pooling` {#id}



Type: `Optional[brevettiai.data.image.annotation_pooling.AnnotationPooling]`




##### Variable `classes` {#id}



Type: `List[str]`




##### Variable `example_image` {#id}



Type: `Optional[brevettiai.model.metadata.image_segmentation.Base64Image]`




##### Variable `image_loader` {#id}



Type: `brevettiai.data.image.image_loader.ImageLoader`




##### Variable `multi_frame_imager` {#id}



Type: `Optional[brevettiai.data.image.multi_frame_imager.MultiFrameImager]`




##### Variable `producer` {#id}



Type: `brevettiai.model.metadata.image_segmentation.ConstrainedStrValue`




##### Variable `suggested_input_shape` {#id}



Type: `Tuple[int, int]`






#### Static methods



##### `Method prepare_input` {#id}




>     def prepare_input(
>         values
>     )








# Module `brevettiai.model.metadata.metadata` {#id}







## Functions



### Function `get_metadata` {#id}




>     def get_metadata(
>         file: str,
>         metadata_type: Type[brevettiai.model.metadata.metadata.ModelMetadata] = brevettiai.model.metadata.metadata.ModelMetadata
>     )






## Classes



### Class `ModelMetadata` {#id}




>     class ModelMetadata(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.model.metadata.image_segmentation.ImageSegmentationModelMetadata](#brevettiai.model.metadata.image_segmentation.ImageSegmentationModelMetadata)



#### Class variables



##### Variable `Config` {#id}







##### Variable `host_name` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `name` {#id}



Type: `str`




##### Variable `producer` {#id}



Type: `str`




##### Variable `run_id` {#id}



Type: `str`









# Module `brevettiai.platform` {#id}





## Sub-modules

* [brevettiai.platform.models](#brevettiai.platform.models)
* [brevettiai.platform.platform_credentials](#brevettiai.platform.platform_credentials)
* [brevettiai.platform.web_api](#brevettiai.platform.web_api)







# Module `brevettiai.platform.models` {#id}





## Sub-modules

* [brevettiai.platform.models.annotation](#brevettiai.platform.models.annotation)
* [brevettiai.platform.models.dataset](#brevettiai.platform.models.dataset)
* [brevettiai.platform.models.job](#brevettiai.platform.models.job)
* [brevettiai.platform.models.platform_backend](#brevettiai.platform.models.platform_backend)
* [brevettiai.platform.models.tag](#brevettiai.platform.models.tag)
* [brevettiai.platform.models.web_api_types](#brevettiai.platform.models.web_api_types)







# Module `brevettiai.platform.models.annotation` {#id}







## Functions



### Function `draw_contours_CHW` {#id}




>     def draw_contours_CHW(
>         annotations,
>         draw_buffer,
>         label_space=None
>     )





### Function `flatten_structure` {#id}




>     def flatten_structure(
>         x,
>         name='',
>         out=None
>     )





### Function `sub_ious` {#id}




>     def sub_ious(
>         annotation,
>         polygons
>     )






## Classes



### Class `Annotation` {#id}




>     class Annotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.platform.models.annotation.ClassAnnotation](#brevettiai.platform.models.annotation.ClassAnnotation)
* [brevettiai.platform.models.annotation.PointsAnnotation](#brevettiai.platform.models.annotation.PointsAnnotation)



#### Class variables



##### Variable `Config` {#id}







##### Variable `color` {#id}



Type: `brevettiai.platform.models.annotation.Color`




##### Variable `features` {#id}



Type: `Dict[str, Any]`




##### Variable `label` {#id}



Type: `str`




##### Variable `severity` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedIntValue`




##### Variable `type` {#id}



Type: `str`




##### Variable `uuid` {#id}



Type: `uuid.UUID`




##### Variable `visibility` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedIntValue`






#### Static methods



##### `Method default_negative_1` {#id}




>     def default_negative_1(
>         v,
>         field
>     )






### Class `ArrayMeta` {#id}




>     class ArrayMeta(
>         *args,
>         **kwargs
>     )


type(object_or_name, bases, dict)
type(object) -> the object's type
type(name, bases, dict) -> a new type



#### Ancestors (in MRO)

* [builtins.type](#builtins.type)







### Class `ClassAnnotation` {#id}




>     class ClassAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.annotation.Annotation](#brevettiai.platform.models.annotation.Annotation)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `type` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedStrValue`







### Class `Color` {#id}




>     class Color(
>         value: Union[Tuple[int, int, int], Tuple[int, int, int, float], str]
>     )


Mixin to provide __str__, __repr__, and __pretty__ methods. See #884 for more details.

__pretty__ is used by [devtools](https://python-devtools.helpmanual.io/) to provide human readable representations
of objects.



#### Ancestors (in MRO)

* [pydantic.color.Color](#pydantic.color.Color)
* [pydantic.utils.Representation](#pydantic.utils.Representation)






#### Static methods



##### `Method from_hsl` {#id}




>     def from_hsl(
>         h,
>         s,
>         l,
>         a=None
>     )






### Class `GroundTruth` {#id}




>     class GroundTruth(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `coverage` {#id}



Type: `float`




##### Variable `iou` {#id}



Type: `float`




##### Variable `label` {#id}



Type: `str`




##### Variable `severity` {#id}



Type: `float`






#### Static methods



##### `Method from_annotation` {#id}




>     def from_annotation(
>         annotation,
>         target=None
>     )






#### Methods



##### Method `dict` {#id}




>     def dict(
>         self,
>         **kwargs
>     )


Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.


### Class `ImageAnnotation` {#id}




>     class ImageAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `annotations` {#id}



Type: `List[Union[brevettiai.platform.models.annotation.PolygonAnnotation, brevettiai.platform.models.annotation.RectangleAnnotation, brevettiai.platform.models.annotation.LineAnnotation, brevettiai.platform.models.annotation.PointAnnotation, brevettiai.platform.models.annotation.ClassAnnotation]]`




##### Variable `image` {#id}



Type: `dict`




##### Variable `source` {#id}



Type: `dict`






#### Static methods



##### `Method extract_features` {#id}




>     def extract_features(
>         channel,
>         features,
>         classes,
>         bbox,
>         mask,
>         threshold,
>         sample,
>         CHW,
>         chierarchy,
>         annotations,
>         annotation,
>         get_features
>     )





##### `Method from_path` {#id}




>     def from_path(
>         path,
>         io=<brevettiai.io.utils.IoTools object>,
>         errors='raise'
>     )





##### `Method from_segmentation` {#id}




>     def from_segmentation(
>         segmentation,
>         classes,
>         sample,
>         image_shape,
>         tform=None,
>         threshold=0.5,
>         simplify=False,
>         output_classes=None,
>         CHW=False,
>         feature_func=None,
>         get_features=None
>     )






#### Methods



##### Method `draw_contours_CHW` {#id}




>     def draw_contours_CHW(
>         self,
>         draw_buffer,
>         label_space=None
>     )





##### Method `fix_invalid` {#id}




>     def fix_invalid(
>         self
>     )





##### Method `intersections` {#id}




>     def intersections(
>         self,
>         right
>     )





##### Method `ious` {#id}




>     def ious(
>         self,
>         right
>     )





##### Method `label_map` {#id}




>     def label_map(
>         self
>     )





##### Method `match_annotations` {#id}




>     def match_annotations(
>         self,
>         b,
>         min_coverage=0.2
>     )





##### Method `to_dataframe` {#id}




>     def to_dataframe(
>         self
>     )





##### Method `transform_annotation` {#id}




>     def transform_annotation(
>         self,
>         matrix
>     )





### Class `LineAnnotation` {#id}




>     class LineAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.annotation.PointsAnnotation](#brevettiai.platform.models.annotation.PointsAnnotation)
* [brevettiai.platform.models.annotation.Annotation](#brevettiai.platform.models.annotation.Annotation)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `type` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedStrValue`







### Class `Mask` {#id}




>     class Mask(
>         ...
>     )


ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using <code>array</code>, <code>zeros</code> or <code>empty</code> (refer
to the See Also section below).  The parameters given here refer to
a low-level method (<code>ndarray(...)</code>) for instantiating an array.

For more information, refer to the <code>numpy</code> module and examine the
methods and attributes of an array.

#### Parameters

(for the __new__ method; see Notes below)

**```shape```** :&ensp;<code>tuple</code> of <code>ints</code>
:   Shape of created array.


**```dtype```** :&ensp;`data-type`, optional
:   Any object that can be interpreted as a numpy data type.


**```buffer```** :&ensp;<code>object exposing buffer interface</code>, optional
:   Used to fill the array with data.


**```offset```** :&ensp;<code>int</code>, optional
:   Offset of array data in buffer.


**```strides```** :&ensp;<code>tuple</code> of <code>ints</code>, optional
:   Strides of data in memory.


**```order```** :&ensp;`{'C', 'F'}`, optional
:   Row-major (C-style) or column-major (Fortran-style) order.

#### Attributes

**```T```** :&ensp;<code>ndarray</code>
:   Transpose of the array.


**```data```** :&ensp;<code>buffer</code>
:   The array's elements, in memory.


**```dtype```** :&ensp;<code>dtype object</code>
:   Describes the format of the elements in the array.


**```flags```** :&ensp;<code>dict</code>
:   Dictionary containing information related to memory use, e.g.,
    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.


**```flat```** :&ensp;<code>numpy.flatiter object</code>
:   Flattened version of the array as an iterator.  The iterator
    allows assignments, e.g., ``x.flat = 3`` (See <code>ndarray.flat</code> for
    assignment examples; TODO).


**```imag```** :&ensp;<code>ndarray</code>
:   Imaginary part of the array.


**```real```** :&ensp;<code>ndarray</code>
:   Real part of the array.


**```size```** :&ensp;<code>int</code>
:   Number of elements in the array.


**```itemsize```** :&ensp;<code>int</code>
:   The memory use of each array element in bytes.


**```nbytes```** :&ensp;<code>int</code>
:   The total number of bytes required to store the array data,
    i.e., ``itemsize * size``.


**```ndim```** :&ensp;<code>int</code>
:   The array's number of dimensions.


**```shape```** :&ensp;<code>tuple</code> of <code>ints</code>
:   Shape of the array.


**```strides```** :&ensp;<code>tuple</code> of <code>ints</code>
:   The step-size required to move from one element to the next in
    memory. For example, a contiguous <code>(3, 4)</code> array of type
    <code>int16</code> in C-order has strides <code>(8, 2)</code>.  This implies that
    to move from element to element in memory requires jumps of 2 bytes.
    To move from row-to-row, one needs to jump 8 bytes at a time
    (``2 * 4``).


**```ctypes```** :&ensp;<code>ctypes object</code>
:   Class containing properties of the array needed for interaction
    with ctypes.


**```base```** :&ensp;<code>ndarray</code>
:   If the array is a view into another array, that array is its <code>base</code>
    (unless that array is also a view).  The <code>base</code> array is where the
    array data is actually stored.

#### See Also

<code>array</code>
:   Construct an array.

<code>zeros</code>
:   Create an array, each element of which is zero.

<code>empty</code>
:   Create an array, but leave its allocated memory unchanged (i.e., it contains "garbage").

<code>dtype</code>
:   Create a data-type.

<code>numpy.typing.NDArray</code>
:   A :term:`generic <generic type>` version of ndarray.

#### Notes

There are two modes of creating an array using <code>\_\_new\_\_</code>:

1. If <code>buffer</code> is None, then only <code>shape</code>, <code>dtype</code>, and <code>order</code>
   are used.
2. If <code>buffer</code> is an object exposing the buffer interface, then
   all keywords are interpreted.

No <code>\_\_init\_\_</code> method is needed because the array is fully initialized
after the <code>\_\_new\_\_</code> method.

#### Examples

These examples illustrate the low-level <code>ndarray</code> constructor.  Refer
to the <code>See Also</code> section above for easier ways of constructing an
ndarray.

First mode, <code>buffer</code> is None:

```python-repl
>>> np.ndarray(shape=(2,2), dtype=float, order='F')
array([[0.0e+000, 0.0e+000], # random
       [     nan, 2.5e-323]])
```


Second mode:

```python-repl
>>> np.ndarray((2,), buffer=np.array([1,2,3]),
...            offset=np.int_().itemsize,
...            dtype=int) # offset = 1*itemsize, i.e. skip first element
array([2, 3])
```




#### Ancestors (in MRO)

* [numpy.ndarray](#numpy.ndarray)






#### Static methods



##### `Method validate_type` {#id}




>     def validate_type(
>         val
>     )






### Class `PointAnnotation` {#id}




>     class PointAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.annotation.PointsAnnotation](#brevettiai.platform.models.annotation.PointsAnnotation)
* [brevettiai.platform.models.annotation.Annotation](#brevettiai.platform.models.annotation.Annotation)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `type` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedStrValue`







#### Methods



##### Method `sample_points` {#id}




>     def sample_points(
>         self,
>         tries=100000
>     )





### Class `PointsAnnotation` {#id}




>     class PointsAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.annotation.Annotation](#brevettiai.platform.models.annotation.Annotation)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.platform.models.annotation.LineAnnotation](#brevettiai.platform.models.annotation.LineAnnotation)
* [brevettiai.platform.models.annotation.PointAnnotation](#brevettiai.platform.models.annotation.PointAnnotation)
* [brevettiai.platform.models.annotation.PolygonAnnotation](#brevettiai.platform.models.annotation.PolygonAnnotation)
* [brevettiai.platform.models.annotation.RectangleAnnotation](#brevettiai.platform.models.annotation.RectangleAnnotation)



#### Class variables



##### Variable `Config` {#id}







##### Variable `ground_truth` {#id}



Type: `brevettiai.platform.models.annotation.GroundTruth`




##### Variable `is_hole` {#id}



Type: `bool`




##### Variable `parent` {#id}



Type: `uuid.UUID`




##### Variable `points` {#id}



Type: `brevettiai.platform.models.annotation.Array`





#### Instance variables



##### Variable `area` {#id}







##### Variable `bbox` {#id}







##### Variable `centroid` {#id}







##### Variable `hu_moments` {#id}







##### Variable `mask` {#id}







##### Variable `moments` {#id}







##### Variable `path_length` {#id}







##### Variable `polygon` {#id}









#### Methods



##### Method `clear_calculated` {#id}




>     def clear_calculated(
>         self
>     )





##### Method `dict` {#id}




>     def dict(
>         self,
>         *args,
>         **kwargs
>     )


Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.


##### Method `fix_polygon` {#id}




>     def fix_polygon(
>         self
>     )





##### Method `flat_features` {#id}




>     def flat_features(
>         self
>     )





##### Method `intersection` {#id}




>     def intersection(
>         self,
>         p2
>     )





##### Method `iou` {#id}




>     def iou(
>         self,
>         p2
>     )





##### Method `sample_points` {#id}




>     def sample_points(
>         self,
>         tries=100000
>     )





##### Method `transform_points` {#id}




>     def transform_points(
>         self,
>         matrix
>     )





### Class `PointsArray` {#id}




>     class PointsArray(
>         ...
>     )


ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using <code>array</code>, <code>zeros</code> or <code>empty</code> (refer
to the See Also section below).  The parameters given here refer to
a low-level method (<code>ndarray(...)</code>) for instantiating an array.

For more information, refer to the <code>numpy</code> module and examine the
methods and attributes of an array.

#### Parameters

(for the __new__ method; see Notes below)

**```shape```** :&ensp;<code>tuple</code> of <code>ints</code>
:   Shape of created array.


**```dtype```** :&ensp;`data-type`, optional
:   Any object that can be interpreted as a numpy data type.


**```buffer```** :&ensp;<code>object exposing buffer interface</code>, optional
:   Used to fill the array with data.


**```offset```** :&ensp;<code>int</code>, optional
:   Offset of array data in buffer.


**```strides```** :&ensp;<code>tuple</code> of <code>ints</code>, optional
:   Strides of data in memory.


**```order```** :&ensp;`{'C', 'F'}`, optional
:   Row-major (C-style) or column-major (Fortran-style) order.

#### Attributes

**```T```** :&ensp;<code>ndarray</code>
:   Transpose of the array.


**```data```** :&ensp;<code>buffer</code>
:   The array's elements, in memory.


**```dtype```** :&ensp;<code>dtype object</code>
:   Describes the format of the elements in the array.


**```flags```** :&ensp;<code>dict</code>
:   Dictionary containing information related to memory use, e.g.,
    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.


**```flat```** :&ensp;<code>numpy.flatiter object</code>
:   Flattened version of the array as an iterator.  The iterator
    allows assignments, e.g., ``x.flat = 3`` (See <code>ndarray.flat</code> for
    assignment examples; TODO).


**```imag```** :&ensp;<code>ndarray</code>
:   Imaginary part of the array.


**```real```** :&ensp;<code>ndarray</code>
:   Real part of the array.


**```size```** :&ensp;<code>int</code>
:   Number of elements in the array.


**```itemsize```** :&ensp;<code>int</code>
:   The memory use of each array element in bytes.


**```nbytes```** :&ensp;<code>int</code>
:   The total number of bytes required to store the array data,
    i.e., ``itemsize * size``.


**```ndim```** :&ensp;<code>int</code>
:   The array's number of dimensions.


**```shape```** :&ensp;<code>tuple</code> of <code>ints</code>
:   Shape of the array.


**```strides```** :&ensp;<code>tuple</code> of <code>ints</code>
:   The step-size required to move from one element to the next in
    memory. For example, a contiguous <code>(3, 4)</code> array of type
    <code>int16</code> in C-order has strides <code>(8, 2)</code>.  This implies that
    to move from element to element in memory requires jumps of 2 bytes.
    To move from row-to-row, one needs to jump 8 bytes at a time
    (``2 * 4``).


**```ctypes```** :&ensp;<code>ctypes object</code>
:   Class containing properties of the array needed for interaction
    with ctypes.


**```base```** :&ensp;<code>ndarray</code>
:   If the array is a view into another array, that array is its <code>base</code>
    (unless that array is also a view).  The <code>base</code> array is where the
    array data is actually stored.

#### See Also

<code>array</code>
:   Construct an array.

<code>zeros</code>
:   Create an array, each element of which is zero.

<code>empty</code>
:   Create an array, but leave its allocated memory unchanged (i.e., it contains "garbage").

<code>dtype</code>
:   Create a data-type.

<code>numpy.typing.NDArray</code>
:   A :term:`generic <generic type>` version of ndarray.

#### Notes

There are two modes of creating an array using <code>\_\_new\_\_</code>:

1. If <code>buffer</code> is None, then only <code>shape</code>, <code>dtype</code>, and <code>order</code>
   are used.
2. If <code>buffer</code> is an object exposing the buffer interface, then
   all keywords are interpreted.

No <code>\_\_init\_\_</code> method is needed because the array is fully initialized
after the <code>\_\_new\_\_</code> method.

#### Examples

These examples illustrate the low-level <code>ndarray</code> constructor.  Refer
to the <code>See Also</code> section above for easier ways of constructing an
ndarray.

First mode, <code>buffer</code> is None:

```python-repl
>>> np.ndarray(shape=(2,2), dtype=float, order='F')
array([[0.0e+000, 0.0e+000], # random
       [     nan, 2.5e-323]])
```


Second mode:

```python-repl
>>> np.ndarray((2,), buffer=np.array([1,2,3]),
...            offset=np.int_().itemsize,
...            dtype=int) # offset = 1*itemsize, i.e. skip first element
array([2, 3])
```




#### Ancestors (in MRO)

* [numpy.ndarray](#numpy.ndarray)



#### Descendants

* [brevettiai.platform.models.annotation.Array](#brevettiai.platform.models.annotation.Array)





#### Static methods



##### `Method from_polygon` {#id}




>     def from_polygon(
>         polygon
>     )





##### `Method validate_type` {#id}




>     def validate_type(
>         val
>     )






#### Methods



##### Method `dict` {#id}




>     def dict(
>         self
>     )





### Class `PolygonAnnotation` {#id}




>     class PolygonAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.annotation.PointsAnnotation](#brevettiai.platform.models.annotation.PointsAnnotation)
* [brevettiai.platform.models.annotation.Annotation](#brevettiai.platform.models.annotation.Annotation)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `type` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedStrValue`







### Class `RectangleAnnotation` {#id}




>     class RectangleAnnotation(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.annotation.PointsAnnotation](#brevettiai.platform.models.annotation.PointsAnnotation)
* [brevettiai.platform.models.annotation.Annotation](#brevettiai.platform.models.annotation.Annotation)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `type` {#id}



Type: `brevettiai.platform.models.annotation.ConstrainedStrValue`





#### Instance variables



##### Variable `contour` {#id}







##### Variable `polygon` {#id}











# Module `brevettiai.platform.models.dataset` {#id}







## Functions



### Function `get_samples` {#id}




>     def get_samples(
>         datasets,
>         target,
>         *args,
>         **kwargs
>     )


Utility function for getting samples across multiple datasets by sample files
:param datasets:
:param target:
:param args:
:param kwargs:
:return:


### Function `load_sample_identification` {#id}




>     def load_sample_identification(
>         df,
>         path,
>         column='purpose',
>         io=<brevettiai.io.utils.IoTools object>,
>         **kwargs
>     )


Load and join sample identification information onto dataframe of samples
:param df: sample dataframe
:param path: path to sample id file
:param column: name of split column
:param kwargs: extra args for io_tools.read_file
:return: df, extra_ids


### Function `save_sample_identification` {#id}




>     def save_sample_identification(
>         df,
>         path,
>         known_ids=None,
>         column='purpose',
>         io=<brevettiai.io.utils.IoTools object>
>     )





### Function `save_samples` {#id}




>     def save_samples(
>         datasets,
>         target,
>         df
>     )





### Function `tif2dzi` {#id}




>     def tif2dzi(
>         path,
>         bucket
>     )






## Classes



### Class `BrevettiDatasetSamples` {#id}




>     class BrevettiDatasetSamples(
>         classes: list = None,
>         class_mapping: dict = None,
>         annotations: dict = None,
>         calculate_md5: bool = False,
>         walk: bool = True,
>         samples_file_name: str = None,
>         contains_column: str = None,
>         contains_regex: str = None
>     )


Base class for serializable modules

:param class_mapping: dict of mapping from path to (category) class. See example for description
:param classes: Force samples to be of the categories in this list



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)







#### Methods



##### Method `get_image_samples` {#id}




>     def get_image_samples(
>         self,
>         datasets,
>         *args,
>         **kwargs
>     )


:param sample_filter: Filter samples by regex
:param annotations: boolean, or dict Load annotation paths
:param kwargs:
:return: Dataframe of samples

Mapping from path to category:
Start from the leaf folder, and work towards the dataset root folder. If folder in class_mapping then apply
its key as the category. If no match is found, apply leaf folder name
Example:
class_mapping={
"A": ["Tilted"],
"B": ["Tilted"],
"1": ["Crimp"]
}
If classes is True or a list/set of categories. The filter is applied after the mapping.


##### Method `get_samples` {#id}




>     def get_samples(
>         self,
>         datasets,
>         walk=None,
>         *args,
>         **kwargs
>     )


Utility function for getting samples across multiple datasets by sample files
:param datasets:
:param target:
:param args:
:param kwargs:
:return:


### Class `SampleSplit` {#id}




>     class SampleSplit(
>         stratification: list = None,
>         uniqueness: list = None,
>         split: float = 0.8,
>         seed: int = -1,
>         mode='sorted_permutation'
>     )


Base class for serializable modules

:param stratification: As regex string performed on df.path or list selecting columns
:param uniqueness: As regex string performed on df.path or list selecting columns
:param split: fraction of samples to apply the purpose on
:param seed: seeding for assignment
:param mode: ' or 'murmurhash3'
:return:



#### Ancestors (in MRO)

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.utils.module.Module](#brevettiai.utils.module.Module)




#### Class variables



##### Variable `MODE_MURMURHASH3` {#id}







##### Variable `MODE_SORTED_PERMUTATION` {#id}










#### Methods



##### Method `assign` {#id}




>     def assign(
>         self,
>         df,
>         purpose='train',
>         remainder=None,
>         column='purpose'
>     )


Assign purpose column randomly to non-assigned samples based on stratification, uniqueness and split strategy.

Definitions:
* Stratification: Grouping of samples which should be treated as individual groups.
meaning every group must be split according to the sample split target percentage,
and uniqueness is performed on a per group basis
* Uniqueness: grouping of samples which must be treated as a single sample, thus be assigned the same purpose.

:param df: pd.DataFrame of samples if purpose column does not exist it is added
:param purpose: purpose to be assigned
:param remainder: purpose to assign remainder samples, or None to leave unassigned
:param column: column for assignment of split category


##### Method `update_unassigned` {#id}




>     def update_unassigned(
>         self,
>         df,
>         id_path,
>         purpose='train',
>         remainder='devel',
>         column='purpose',
>         io=<brevettiai.io.utils.IoTools object>
>     )


Updates sample purpose in id_path that may hold previous dataset splits and sample ids
Unassigned samples are also assigned and id_path is updated
:param df: pd.DataFrame containing the samples
:param id_path: path to the identification csv file
:param purpose: Purpose to assign
:param remainder: Purpose to assign to remainder or none to leave unassigned
:param column: Column to assign split purposes to
:return:




# Module `brevettiai.platform.models.job` {#id}







## Functions



### Function `parse_job_args` {#id}




>     def parse_job_args()






## Classes



### Class `Job` {#id}




>     class Job(
>         io=<brevettiai.io.utils.IoTools object>,
>         backend=PlatformBackend(host='https://platform.brevetti.ai', output_segmentation_dir='output_segmentations', bucket_region='eu-west-1', data_bucket='s3://data.criterion.ai', custom_job_id='a0aaad69-c032-41c1-a68c-e9a15a5fb18c'),
>         cache_path=None,
>         job_dir=None,
>         **data
>     )


Model defining a job on the Brevetti platform,
Use it as base class for your jobs

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `Config` {#id}







##### Variable `api_key` {#id}



Type: `Optional[str]`




##### Variable `charts_url` {#id}



Type: `Optional[str]`




##### Variable `complete_url` {#id}



Type: `Optional[str]`




##### Variable `datasets` {#id}



Type: `List[brevettiai.platform.models.dataset.Dataset]`




##### Variable `host_name` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `model_path` {#id}



Type: `Optional[str]`




##### Variable `models` {#id}



Type: `List[dict]`




##### Variable `name` {#id}



Type: `str`




##### Variable `parent` {#id}



Type: `Optional[dict]`




##### Variable `remote_url` {#id}



Type: `Optional[str]`




##### Variable `run_id` {#id}



Type: `str`




##### Variable `security_credentials_url` {#id}



Type: `Optional[str]`




##### Variable `settings` {#id}



Type: `brevettiai.platform.models.job.JobSettings`




##### Variable `tags` {#id}



Type: `List[brevettiai.platform.models.tag.Tag]`





#### Instance variables



##### Variable `backend` {#id}




Reference to the backend


##### Variable `io` {#id}




Io reference for file handling


##### Variable `job_dir` {#id}



Type: `str`





#### Static methods



##### `Method from_model_spec` {#id}




>     def from_model_spec(
>         model,
>         schema=None,
>         config=None,
>         job_dir=None,
>         **kwargs
>     )


Build job object from model specification
:param model:
:param schema:
:param config:
:param kwargs:
:param job_dir:
:return:


##### `Method init` {#id}




>     def init(
>         job_id: Optional[str] = None,
>         api_key: Optional[str] = None,
>         cache_path: Optional[str] = None,
>         info_file: Optional[str] = None,
>         type_selector: Union[Iterable[Type[ForwardRef('Job')]], Callable[[dict], Type[ForwardRef('Job')]]] = None,
>         job_config: Optional[dict] = None,
>         log_level=20,
>         io=<brevettiai.io.utils.IoTools object>,
>         backend=PlatformBackend(host='https://platform.brevetti.ai', output_segmentation_dir='output_segmentations', bucket_region='eu-west-1', data_bucket='s3://data.criterion.ai', custom_job_id='a0aaad69-c032-41c1-a68c-e9a15a5fb18c'),
>         **kwargs
>     ) ‑> brevettiai.platform.models.job.Job


Initialize a job
:param job_id: id of job to find on the backend
:param api_key: Api key for access if job is containing remote resources
:param info_file: filename of info file to use, overwrites job id
:param cache_path: Path to use for caching remote resources and as temporary storage
:param type_selector: list of different job types to match
:param job_config: configuration of the job
:param log_level: logging level
:param io IoTools: object managing data access and reads / writes
:param backend: PlatformBackend object containing info on the backend to use



#### Methods



##### Method `add_output_metric` {#id}




>     def add_output_metric(
>         self,
>         key,
>         metric
>     )


Add an output metric for the job comparison module
:param key:
:param metric:
:return:


##### Method `add_output_metrics` {#id}




>     def add_output_metrics(
>         self,
>         metrics
>     )


Add a number of metrics to the job
:param metrics:
:return:


##### Method `artifact_path` {#id}




>     def artifact_path(
>         self,
>         *paths,
>         dir: bool = False
>     ) ‑> str


Get path in the artifact directory tree
:param paths: N path arguments
:param dir: this is a directory
:return:


##### Method `complete` {#id}




>     def complete(
>         self,
>         tmp_package_path=None,
>         package_path=None,
>         output_args=''
>     )


Complete job by uploading package to gcs and notifying api
:param tmp_package_path: Path to tar archive with python package
:param package_path: package path on gcs
:param output_args
:return:


##### Method `get_annotation_url` {#id}




>     def get_annotation_url(
>         self,
>         s3_image_path,
>         annotation_name=None,
>         bbox=None,
>         zoom=None,
>         screen_size=1024,
>         test_report_id=None,
>         model_id=None,
>         min_zoom=2,
>         max_zoom=300
>     )


Get url to annotation file
:param s3_image_path: Name of image file
:param annotation_name: Name of annotation file, if any
:param bbox: Selects zoom and center for the bbox
:param zoom: Zoom level [2-300] related to screen pixel size (if None zoom will be calculated from bbox)
:param screen_size: default screen size in pixels
:param test_report_id:
:param model_id:
:param min_zoom:
:param max_zoom:


##### Method `get_metadata` {#id}




>     def get_metadata(
>         self
>     ) ‑> brevettiai.model.metadata.metadata.ModelMetadata


Build metadata object, containing information to transfer to external users of the model


##### Method `get_remote_monitor` {#id}




>     def get_remote_monitor(
>         self
>     )


Retrieve remote monitor object if available


##### Method `prepare_path` {#id}




>     def prepare_path(
>         self,
>         *paths,
>         dir: bool = False
>     ) ‑> str





##### Method `resolve_access_rights` {#id}




>     def resolve_access_rights(
>         self
>     ) ‑> None


Resolve access rights of this job
:return:


##### Method `run` {#id}




>     def run(
>         self
>     ) ‑> Optional[str]


Overwrite this to run your job
Return path to model in temp dir to upload


##### Method `start` {#id}




>     def start(
>         self,
>         resolve_access_rights: bool = True,
>         cache_remote_files: bool = True,
>         set_credentials: bool = True,
>         complete_job: bool = True
>     )


Start the job


##### Method `temp_path` {#id}




>     def temp_path(
>         self,
>         *paths,
>         dir=False
>     )


Get path in the temp directory tree
:param paths: N path arguments
:param dir: this is a directory
:return:


##### Method `upload_artifact` {#id}




>     def upload_artifact(
>         self,
>         artifact_name,
>         payload,
>         is_file=False
>     )


Upload an artifact with a given name
:param artifact_name: target artifact name
:param payload: source
:param is_file: payload is string to file location
:return:


##### Method `upload_job_output` {#id}




>     def upload_job_output(
>         self,
>         path='output.json'
>     )


Upload / update the output.json artifact containing parsed settings, etc.
:param path:
:return:


### Class `JobOutput` {#id}




>     class JobOutput(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `Config` {#id}







##### Variable `environment` {#id}



Type: `Dict[str, Optional[str]]`




##### Variable `job` {#id}



Type: `brevettiai.platform.models.job.Job`




##### Variable `output` {#id}



Type: `dict`







### Class `JobSettings` {#id}




>     class JobSettings(
>         **data: Any
>     )


Baseclass for job settings

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `Config` {#id}







##### Variable `extra` {#id}



Type: `Dict[str, Any]`






#### Static methods



##### `Method parse_settings` {#id}




>     def parse_settings(
>         values: Dict[str, Any]
>     ) ‑> Dict[str, Any]


Extra validator parsing settings from the platform


##### `Method platform_schema` {#id}




>     def platform_schema()


Utility function to get vue schema





# Module `brevettiai.platform.models.platform_backend` {#id}








## Classes



### Class `PlatformBackend` {#id}




>     class PlatformBackend(
>         **data: Any
>     )


Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `bucket_region` {#id}



Type: `str`




##### Variable `custom_job_id` {#id}



Type: `str`




##### Variable `data_bucket` {#id}



Type: `str`




##### Variable `host` {#id}



Type: `str`




##### Variable `output_segmentation_dir` {#id}



Type: `str`





#### Instance variables



##### Variable `custom_model_type` {#id}







##### Variable `s3_endpoint` {#id}









#### Methods



##### Method `get_annotation_url` {#id}




>     def get_annotation_url(
>         self,
>         s3_image_path,
>         annotation_name=None,
>         bbox=None,
>         zoom=None,
>         screen_size=1024,
>         test_report_id=None,
>         model_id=None,
>         min_zoom=2,
>         max_zoom=300
>     )


Get url to annotation file
:param s3_image_path: Name of image file
:param annotation_name: Name of annotation file, if any
:param bbox: Selects zoom and center for the bbox
:param zoom: Zoom level [2-300] related to screen pixel size (if None zoom will be calculated from bbox)
:param screen_size: default screen size in pixels
:param test_report_id:
:param model_id:
:param min_zoom:
:param max_zoom:


##### Method `get_download_link` {#id}




>     def get_download_link(
>         self,
>         path
>     )





##### Method `get_root_tags` {#id}




>     def get_root_tags(
>         self,
>         id,
>         api_key
>     ) ‑> List[brevettiai.platform.models.tag.Tag]





##### Method `prepare_runtime` {#id}




>     def prepare_runtime(
>         self
>     )





##### Method `resource_path` {#id}




>     def resource_path(
>         self,
>         uuid: Union[str, uuid.UUID]
>     ) ‑> str


Get location of a resource




# Module `brevettiai.platform.models.tag` {#id}








## Classes



### Class `Tag` {#id}




>     class Tag(
>         **data: Any
>     )


Model defining a Tag on the Brevetti platform

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `Config` {#id}







##### Variable `children` {#id}



Type: `List[brevettiai.platform.models.tag.Tag]`




##### Variable `created` {#id}



Type: `str`




##### Variable `id` {#id}



Type: `str`




##### Variable `name` {#id}



Type: `str`




##### Variable `parent_id` {#id}



Type: `Optional[str]`






#### Static methods



##### `Method find` {#id}




>     def find(
>         tree,
>         key,
>         value
>     )





##### `Method find_path` {#id}




>     def find_path(
>         tree,
>         key,
>         value,
>         path=()
>     )








# Module `brevettiai.platform.models.web_api_types` {#id}







## Functions



### Function `to_camel` {#id}




>     def to_camel(
>         x
>     )






## Classes



### Class `Application` {#id}




>     class Application(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `created` {#id}



Type: `str`




##### Variable `description` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `model_ids` {#id}



Type: `List[str]`




##### Variable `name` {#id}



Type: `str`




##### Variable `starred_model_ids` {#id}



Type: `List[str]`




##### Variable `test_dataset_ids` {#id}



Type: `List[str]`




##### Variable `thumbnail_data` {#id}



Type: `Optional[str]`




##### Variable `training_dataset_ids` {#id}



Type: `List[str]`




##### Variable `type` {#id}



Type: `int`





#### Instance variables



##### Variable `related_ids` {#id}









### Class `CamelModel` {#id}




>     class CamelModel(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)



#### Descendants

* [brevettiai.platform.models.web_api_types.Application](#brevettiai.platform.models.web_api_types.Application)
* [brevettiai.platform.models.web_api_types.Device](#brevettiai.platform.models.web_api_types.Device)
* [brevettiai.platform.models.web_api_types.FileEntry](#brevettiai.platform.models.web_api_types.FileEntry)
* [brevettiai.platform.models.web_api_types.Model](#brevettiai.platform.models.web_api_types.Model)
* [brevettiai.platform.models.web_api_types.ModelType](#brevettiai.platform.models.web_api_types.ModelType)
* [brevettiai.platform.models.web_api_types.Permission](#brevettiai.platform.models.web_api_types.Permission)
* [brevettiai.platform.models.web_api_types.Project](#brevettiai.platform.models.web_api_types.Project)
* [brevettiai.platform.models.web_api_types.Report](#brevettiai.platform.models.web_api_types.Report)
* [brevettiai.platform.models.web_api_types.ReportType](#brevettiai.platform.models.web_api_types.ReportType)
* [brevettiai.platform.models.web_api_types.SftpUser](#brevettiai.platform.models.web_api_types.SftpUser)
* [brevettiai.platform.models.web_api_types.User](#brevettiai.platform.models.web_api_types.User)
* [brevettiai.platform.models.web_api_types.UserPermissions](#brevettiai.platform.models.web_api_types.UserPermissions)



#### Class variables



##### Variable `Config` {#id}










### Class `Device` {#id}




>     class Device(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `actual_configuration` {#id}



Type: `Optional[str]`




##### Variable `applications` {#id}



Type: `List[brevettiai.platform.models.web_api_types.Application]`




##### Variable `created` {#id}



Type: `str`




##### Variable `datasets` {#id}



Type: `Optional[List[brevettiai.platform.models.dataset.Dataset]]`




##### Variable `deployments` {#id}



Type: `Optional[List[dict]]`




##### Variable `desired_configuration` {#id}



Type: `Optional[str]`




##### Variable `firmware_version` {#id}



Type: `str`




##### Variable `id` {#id}



Type: `str`




##### Variable `name` {#id}



Type: `str`




##### Variable `password` {#id}



Type: `Optional[str]`




##### Variable `tag_ids` {#id}



Type: `List[str]`







### Class `FileEntry` {#id}




>     class FileEntry(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `is_prefix` {#id}



Type: `bool`




##### Variable `last_modified` {#id}



Type: `Optional[str]`




##### Variable `link` {#id}



Type: `Optional[str]`




##### Variable `mime_type` {#id}



Type: `str`




##### Variable `name` {#id}



Type: `str`




##### Variable `size` {#id}



Type: `Optional[int]`




##### Variable `tile_source` {#id}



Type: `Optional[str]`







### Class `Model` {#id}




>     class Model(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `api_key` {#id}



Type: `str`




##### Variable `application_id` {#id}



Type: `Optional[str]`




##### Variable `completed` {#id}



Type: `Optional[str]`




##### Variable `created` {#id}



Type: `str`




##### Variable `dataset_ids` {#id}



Type: `Optional[List[str]]`




##### Variable `has_deployments` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `job_id` {#id}



Type: `Optional[str]`




##### Variable `model_type_id` {#id}



Type: `Optional[str]`




##### Variable `model_type_status` {#id}



Type: `int`




##### Variable `model_url` {#id}



Type: `Optional[str]`




##### Variable `name` {#id}



Type: `str`




##### Variable `report_type_ids` {#id}



Type: `Optional[List[str]]`




##### Variable `settings` {#id}



Type: `str`




##### Variable `started` {#id}



Type: `Optional[str]`




##### Variable `tag_ids` {#id}



Type: `Optional[List[str]]`




##### Variable `version` {#id}



Type: `Optional[str]`





#### Instance variables



##### Variable `has_api_key` {#id}









#### Methods



##### Method `get_datasets` {#id}




>     def get_datasets(
>         self,
>         api
>     )





### Class `ModelType` {#id}




>     class ModelType(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `created` {#id}



Type: `str`




##### Variable `docker_image` {#id}



Type: `Optional[str]`




##### Variable `duplicate_of` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `instance_count` {#id}



Type: `int`




##### Variable `instance_type` {#id}



Type: `Optional[str]`




##### Variable `long_description` {#id}



Type: `Optional[str]`




##### Variable `max_runtime_in_seconds` {#id}



Type: `int`




##### Variable `name` {#id}



Type: `str`




##### Variable `organization` {#id}



Type: `Optional[str]`




##### Variable `organization_id` {#id}



Type: `Optional[str]`




##### Variable `report_type_ids` {#id}



Type: `List[str]`




##### Variable `report_types` {#id}



Type: `Optional[str]`




##### Variable `settings_schema_name` {#id}



Type: `Optional[str]`




##### Variable `settings_schema_path` {#id}



Type: `Optional[str]`




##### Variable `short_description` {#id}



Type: `Optional[str]`




##### Variable `status` {#id}



Type: `int`




##### Variable `version` {#id}



Type: `Optional[str]`




##### Variable `volume_size_in_gb` {#id}



Type: `int`







### Class `Permission` {#id}




>     class Permission(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `by_group` {#id}



Type: `bool`




##### Variable `granted` {#id}



Type: `str`




##### Variable `id` {#id}



Type: `str`




##### Variable `role` {#id}



Type: `str`







### Class `Project` {#id}




>     class Project(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `applications` {#id}



Type: `List[brevettiai.platform.models.web_api_types.Application]`




##### Variable `created` {#id}



Type: `str`




##### Variable `description` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `name` {#id}



Type: `str`




##### Variable `thumbnail_data` {#id}



Type: `Optional[str]`







### Class `Report` {#id}




>     class Report(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `api_key` {#id}



Type: `str`




##### Variable `completed` {#id}



Type: `Optional[str]`




##### Variable `created` {#id}



Type: `str`




##### Variable `dataset_ids` {#id}



Type: `Optional[List[str]]`




##### Variable `has_deployments` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `job_id` {#id}



Type: `Optional[str]`




##### Variable `model_ids` {#id}



Type: `Optional[List[str]]`




##### Variable `name` {#id}



Type: `str`




##### Variable `parent_id` {#id}



Type: `str`




##### Variable `parent_name` {#id}



Type: `Optional[str]`




##### Variable `parent_type` {#id}



Type: `str`




##### Variable `parent_version` {#id}



Type: `Optional[str]`




##### Variable `project_id` {#id}



Type: `Optional[str]`




##### Variable `report_type_id` {#id}



Type: `str`




##### Variable `report_type_name` {#id}



Type: `Optional[str]`




##### Variable `report_type_status` {#id}



Type: `int`




##### Variable `report_type_version` {#id}



Type: `Optional[str]`




##### Variable `settings` {#id}



Type: `str`




##### Variable `started` {#id}



Type: `Optional[str]`




##### Variable `tag_ids` {#id}



Type: `Optional[List[str]]`





#### Instance variables



##### Variable `has_api_key` {#id}









#### Methods



##### Method `get_datasets` {#id}




>     def get_datasets(
>         self,
>         api
>     )





### Class `ReportType` {#id}




>     class ReportType(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `can_run_on_applications` {#id}



Type: `bool`




##### Variable `can_run_on_models` {#id}



Type: `bool`




##### Variable `can_run_on_projects` {#id}



Type: `bool`




##### Variable `created` {#id}



Type: `str`




##### Variable `docker_image` {#id}



Type: `Optional[str]`




##### Variable `duplicate_of` {#id}



Type: `Optional[str]`




##### Variable `id` {#id}



Type: `str`




##### Variable `instance_count` {#id}



Type: `int`




##### Variable `instance_type` {#id}



Type: `Optional[str]`




##### Variable `long_description` {#id}



Type: `Optional[str]`




##### Variable `max_runtime_in_seconds` {#id}



Type: `int`




##### Variable `model_type_ids` {#id}



Type: `List[str]`




##### Variable `model_types` {#id}



Type: `Optional[str]`




##### Variable `name` {#id}



Type: `str`




##### Variable `organization` {#id}



Type: `Optional[str]`




##### Variable `organization_id` {#id}



Type: `Optional[str]`




##### Variable `settings_schema_name` {#id}



Type: `Optional[str]`




##### Variable `settings_schema_path` {#id}



Type: `Optional[str]`




##### Variable `short_description` {#id}



Type: `Optional[str]`




##### Variable `status` {#id}



Type: `int`




##### Variable `version` {#id}



Type: `Optional[str]`




##### Variable `volume_size_in_gb` {#id}



Type: `int`







### Class `SftpUser` {#id}




>     class SftpUser(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `folder` {#id}



Type: `Optional[str]`




##### Variable `private_key` {#id}



Type: `Optional[str]`




##### Variable `public_key` {#id}



Type: `Optional[str]`




##### Variable `user_name` {#id}



Type: `str`







### Class `User` {#id}




>     class User(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `accepts_transactional_emails` {#id}



Type: `bool`




##### Variable `api_key` {#id}



Type: `str`




##### Variable `email` {#id}



Type: `str`




##### Variable `first_name` {#id}



Type: `str`




##### Variable `has_access_to_master_mode` {#id}



Type: `bool`




##### Variable `has_password` {#id}



Type: `bool`




##### Variable `id` {#id}



Type: `str`




##### Variable `is_admin` {#id}



Type: `bool`




##### Variable `is_admin_or_power_user` {#id}



Type: `bool`




##### Variable `is_email_confirmed` {#id}



Type: `bool`




##### Variable `last_name` {#id}



Type: `str`




##### Variable `must_authenticate_externally` {#id}



Type: `bool`




##### Variable `permissions` {#id}



Type: `brevettiai.platform.models.web_api_types.UserPermissions`




##### Variable `phone_number` {#id}



Type: `Optional[str]`




##### Variable `plan` {#id}



Type: `int`




##### Variable `status_message` {#id}



Type: `Optional[str]`




##### Variable `username` {#id}



Type: `str`







### Class `UserPermissions` {#id}




>     class UserPermissions(
>         **data: Any
>     )


Model with camel cased aliases for all fields by default

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [brevettiai.platform.models.web_api_types.CamelModel](#brevettiai.platform.models.web_api_types.CamelModel)
* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `datasets` {#id}



Type: `List[brevettiai.platform.models.web_api_types.Permission]`




##### Variable `devices` {#id}



Type: `List[brevettiai.platform.models.web_api_types.Permission]`




##### Variable `models` {#id}



Type: `List[brevettiai.platform.models.web_api_types.Permission]`




##### Variable `projects` {#id}



Type: `List[brevettiai.platform.models.web_api_types.Permission]`









# Module `brevettiai.platform.platform_credentials` {#id}








## Classes



### Class `DefaultJobCredentialsChain` {#id}




>     class DefaultJobCredentialsChain(
>         chain: List[brevettiai.io.credentials.Credentials] = <factory>
>     )


Default credentials chain for jobs, using api keys, AWS configuration and then Sagemaker as source of login



#### Ancestors (in MRO)

* [brevettiai.io.credentials.CredentialsChain](#brevettiai.io.credentials.CredentialsChain)
* [brevettiai.io.credentials.Credentials](#brevettiai.io.credentials.Credentials)
* [abc.ABC](#abc.ABC)




#### Class variables



##### Variable `chain` {#id}



Type: `List[brevettiai.io.credentials.Credentials]`







### Class `JobCredentials` {#id}




>     class JobCredentials(
>         guid=None,
>         apiKey=None,
>         host=None
>     )


Credentials manager for the job context



#### Ancestors (in MRO)

* [brevettiai.io.credentials.Credentials](#brevettiai.io.credentials.Credentials)
* [abc.ABC](#abc.ABC)





#### Instance variables



##### Variable `platform` {#id}









#### Methods



##### Method `get_credentials` {#id}




>     def get_credentials(
>         self,
>         resource_id,
>         resource_type='dataset',
>         mode='r'
>     )





##### Method `get_sts_access_url` {#id}




>     def get_sts_access_url(
>         self,
>         resource_id,
>         resource_type,
>         mode
>     )


get url for requesting sts token
:param resource_id: id of resource
:param resource_type: type of resource 'dataset', 'job'
:param mode: 'read' / 'r', 'write' / 'w'
:return:


##### Method `get_sts_credentials` {#id}




>     def get_sts_credentials(
>         self,
>         resource_id,
>         resource_type,
>         mode
>     )





##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         type,
>         user,
>         secret,
>         platform='__keep__',
>         **kwargs
>     )


Set api credentials to use
:param type: reacts if type is 'JobCredentials'
:param user: the job GUID
:param secret: the job apiKey
:param platform:
:return:


### Class `PlatformDatasetCredentials` {#id}




>     class PlatformDatasetCredentials(
>         platform_api: PlatformAPI
>     )


Credentials manager for platform users



#### Ancestors (in MRO)

* [brevettiai.io.credentials.Credentials](#brevettiai.io.credentials.Credentials)
* [abc.ABC](#abc.ABC)




#### Class variables



##### Variable `platform_api` {#id}



Type: `PlatformAPI`







#### Methods



##### Method `get_credentials` {#id}




>     def get_credentials(
>         self,
>         resource_id,
>         resource_type='dataset',
>         mode='r'
>     )





##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         type,
>         user,
>         secret,
>         **kwargs
>     )







# Module `brevettiai.platform.web_api` {#id}








## Classes



### Class `PlatformAPI` {#id}




>     class PlatformAPI(
>         username=None,
>         password=None,
>         host=None,
>         remember_me=False
>     )









#### Instance variables



##### Variable `antiforgery_headers` {#id}




Get anti forgery headers from platform
:return:


##### Variable `backend` {#id}







##### Variable `host` {#id}







##### Variable `io` {#id}









#### Methods



##### Method `create` {#id}




>     def create(
>         self,
>         obj: Union[brevettiai.platform.models.dataset.Dataset, brevettiai.platform.models.tag.Tag, brevettiai.platform.models.web_api_types.Model, brevettiai.platform.models.web_api_types.Report],
>         **kwargs
>     )





##### Method `create_model` {#id}




>     def create_model(
>         self,
>         name,
>         datasets,
>         settings: brevettiai.platform.models.job.JobSettings = None,
>         model_type=None,
>         tags=None,
>         application: brevettiai.platform.models.web_api_types.Application = None
>     )


Create new model
:param name:
:param model_type:
:param settings:
:param datasets:
:param tags:
:param application:
:return:


##### Method `create_sftp_user` {#id}




>     def create_sftp_user(
>         self,
>         dataset,
>         **kwargs
>     ) ‑> brevettiai.platform.models.web_api_types.SftpUser





##### Method `create_testreport` {#id}




>     def create_testreport(
>         self,
>         name,
>         model,
>         datasets,
>         report_type,
>         settings,
>         tags,
>         submitToCloud=False
>     )


Create new test report
:param name:
:param model:
:param datasets:
:param report_type:
:param settings:
:param tags:
:param submitToCloud:
:return:


##### Method `delete` {#id}




>     def delete(
>         self,
>         obj: Union[brevettiai.platform.models.dataset.Dataset, brevettiai.platform.models.tag.Tag, brevettiai.platform.models.web_api_types.Model, brevettiai.platform.models.web_api_types.Report, brevettiai.platform.models.web_api_types.SftpUser]
>     )





##### Method `download_url` {#id}




>     def download_url(
>         self,
>         url,
>         dst=None,
>         headers=None
>     )





##### Method `get_application` {#id}




>     def get_application(
>         self,
>         id=None
>     ) ‑> Union[brevettiai.platform.models.web_api_types.Application, List[brevettiai.platform.models.web_api_types.Application]]


Get application by id
:param id: either application id or associated id (model, dataset)
:return:


##### Method `get_artifacts` {#id}




>     def get_artifacts(
>         self,
>         obj: Union[brevettiai.platform.models.web_api_types.Model, brevettiai.platform.models.web_api_types.Report],
>         prefix: str = ''
>     )


Get artifacts for model or test report
:param obj: model/test report object
:param prefix: object prefix (folder)
:return:


##### Method `get_available_model_types` {#id}




>     def get_available_model_types(
>         self
>     )


List all available model types
:return:


##### Method `get_dataset` {#id}




>     def get_dataset(
>         self,
>         id=None,
>         write_access=False,
>         **kwargs
>     ) ‑> Union[brevettiai.platform.models.dataset.Dataset, List[brevettiai.platform.models.dataset.Dataset]]


Get dataset, or list of all datasets
:param id: guid of dataset (accessible from url on platform) or None for all dataset
:param write_access: Assume read and write access to dataset
:param kwargs: Extended search criteria: use ('name', 'reference' 'locked', ...)
:return:


##### Method `get_dataset_sts_assume_role_response` {#id}




>     def get_dataset_sts_assume_role_response(
>         self,
>         guid
>     )





##### Method `get_device` {#id}




>     def get_device(
>         self,
>         id=None
>     )





##### Method `get_model` {#id}




>     def get_model(
>         self,
>         id=None,
>         **kwargs
>     ) ‑> Union[brevettiai.platform.models.web_api_types.Model, List[brevettiai.platform.models.web_api_types.Model]]


Get model or list of all models
:param id: Guid of model (available in the url), or None
:param kwargs: Extended search criteria: use ('name', ...)
:return:


##### Method `get_modeltype` {#id}




>     def get_modeltype(
>         self,
>         id=None,
>         master=False
>     ) ‑> Union[brevettiai.platform.models.web_api_types.ModelType, List[brevettiai.platform.models.web_api_types.ModelType]]


Grt type of model
:param id: model guid
:param master: get from master
:return:


##### Method `get_project` {#id}




>     def get_project(
>         self,
>         id=None
>     ) ‑> Union[brevettiai.platform.models.web_api_types.Project, List[brevettiai.platform.models.web_api_types.Project]]





##### Method `get_report` {#id}




>     def get_report(
>         self,
>         id=None,
>         **kwargs
>     ) ‑> Union[brevettiai.platform.models.web_api_types.Report, List[brevettiai.platform.models.web_api_types.Report]]


Get test report, or list of all reports
:param id: Guid of test report (available in the url), or None
:param kwargs: Extended search criteria: use ('name', ...)
:return:


##### Method `get_reporttype` {#id}




>     def get_reporttype(
>         self,
>         id=None,
>         master=False
>     ) ‑> Union[brevettiai.platform.models.web_api_types.ReportType, List[brevettiai.platform.models.web_api_types.ReportType]]


Grt type of model
:param id: model guid
:param master: get from master
:return:


##### Method `get_schema` {#id}




>     def get_schema(
>         self,
>         obj: Union[brevettiai.platform.models.web_api_types.ModelType, brevettiai.platform.models.web_api_types.ReportType]
>     )


Get schema for a certain model type
:param obj modeltype or report type
:return:


##### Method `get_sftp_users` {#id}




>     def get_sftp_users(
>         self,
>         dataset,
>         **kwargs
>     ) ‑> List[brevettiai.platform.models.web_api_types.SftpUser]





##### Method `get_tag` {#id}




>     def get_tag(
>         self,
>         id=None
>     ) ‑> Union[brevettiai.platform.models.tag.Tag, List[brevettiai.platform.models.tag.Tag]]


Get tag or list of all tags
:param id: tag guid
:return:


##### Method `get_userinfo` {#id}




>     def get_userinfo(
>         self
>     )


Get info on user
:return:


##### Method `initialize_training` {#id}




>     def initialize_training(
>         self,
>         model: Union[str, brevettiai.platform.models.web_api_types.Model],
>         job_type: Type[brevettiai.platform.models.job.Job] = None,
>         submitToCloud=False
>     ) ‑> Optional[brevettiai.platform.models.job.Job]


Start training flow
:param model: model or model id
:param job_type:
:param submitToCloud: submit training to the cloud
:return: updated model


##### Method `login` {#id}




>     def login(
>         self,
>         username,
>         password,
>         remember_me=False
>     )





##### Method `stop_model_training` {#id}




>     def stop_model_training(
>         self,
>         model
>     )


Stop training flow
:param model: model
:return: updated model


##### Method `update` {#id}




>     def update(
>         self,
>         obj,
>         master=False
>     )





##### Method `update_dataset_permission` {#id}




>     def update_dataset_permission(
>         self,
>         id,
>         user_id,
>         group_id=None,
>         permission_type='Editor'
>     )


Update dataset permissions for user
:param id:
:param user_id:
:param group_id:
:param permission_type:
:return:


### Class `WebApiConfig` {#id}




>     class WebApiConfig(
>         **data: Any
>     )


Keeps track of web api setup

Create a new model by parsing and validating input data from keyword arguments.

Raises ValidationError if the input data cannot be parsed to form a valid model.



#### Ancestors (in MRO)

* [pydantic.main.BaseModel](#pydantic.main.BaseModel)
* [pydantic.utils.Representation](#pydantic.utils.Representation)




#### Class variables



##### Variable `secret` {#id}



Type: `bytes`





#### Instance variables



##### Variable `is_modified` {#id}




Is the configuration modified?



#### Static methods



##### `Method load` {#id}




>     def load()


Load WebApiConfig from config_file



#### Methods



##### Method `get_credentials` {#id}




>     def get_credentials(
>         self
>     )


Get Username and password for platform login
:return: username, password


##### Method `save` {#id}




>     def save(
>         self
>     )


Save WebApiConfig to config_file


##### Method `set_credentials` {#id}




>     def set_credentials(
>         self,
>         username: str,
>         password: str
>     )


Set credentials for later retrieval




# Module `brevettiai.tests` {#id}





## Sub-modules

* [brevettiai.tests.test_data](#brevettiai.tests.test_data)
* [brevettiai.tests.test_data_image](#brevettiai.tests.test_data_image)
* [brevettiai.tests.test_data_manipulation](#brevettiai.tests.test_data_manipulation)
* [brevettiai.tests.test_image_loader](#brevettiai.tests.test_image_loader)
* [brevettiai.tests.test_model_metadata](#brevettiai.tests.test_model_metadata)
* [brevettiai.tests.test_pivot](#brevettiai.tests.test_pivot)
* [brevettiai.tests.test_platform_job](#brevettiai.tests.test_platform_job)
* [brevettiai.tests.test_polygon_extraction](#brevettiai.tests.test_polygon_extraction)
* [brevettiai.tests.test_schema](#brevettiai.tests.test_schema)
* [brevettiai.tests.test_tags](#brevettiai.tests.test_tags)




## Functions



### Function `get_resource` {#id}




>     def get_resource(
>         path
>     )


Get resource in tests/bin/





# Module `brevettiai.tests.test_data` {#id}








## Classes



### Class `TestDataGenerator` {#id}




>     class TestDataGenerator(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `n` {#id}







##### Variable `samples` {#id}










#### Methods



##### Method `test_repeated` {#id}




>     def test_repeated(
>         self,
>         repeat=2
>     )





##### Method `test_sample_weighing` {#id}




>     def test_sample_weighing(
>         self
>     )





##### Method `test_sample_weighing_unshuffled` {#id}




>     def test_sample_weighing_unshuffled(
>         self
>     )





##### Method `test_shuffle_repeated` {#id}




>     def test_shuffle_repeated(
>         self,
>         repeat=2
>     )





##### Method `test_shuffled` {#id}




>     def test_shuffled(
>         self,
>         batch_size=3
>     )





##### Method `test_unshuffled_batched` {#id}




>     def test_unshuffled_batched(
>         self,
>         batch_size=3
>     )





##### Method `test_unshuffled_unbatched` {#id}




>     def test_unshuffled_unbatched(
>         self
>     )





### Class `TestDataPurposeAssignment` {#id}




>     class TestDataPurposeAssignment(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `samples` {#id}










#### Methods



##### Method `assertAlmostEqual` {#id}




>     def assertAlmostEqual(
>         self,
>         first: float,
>         second: float,
>         delta,
>         *args,
>         **kwargs
>     ) ‑> None


Fail if the two objects are unequal as determined by their
difference rounded to the given number of decimal places
(default 7) and comparing to zero, or by comparing that the
difference between the two objects is more than the given
delta.

Note that decimal places (from zero) are usually not the same
as significant digits (measured from the most significant digit).

If the two objects compare equal then they will automatically
compare almost equal.


##### Method `test_basic_assignment` {#id}




>     def test_basic_assignment(
>         self
>     )





##### Method `test_basic_assignment_mmh3` {#id}




>     def test_basic_assignment_mmh3(
>         self
>     )





##### Method `test_no_data` {#id}




>     def test_no_data(
>         self
>     )





##### Method `test_stratification` {#id}




>     def test_stratification(
>         self
>     )





##### Method `test_uniqueness` {#id}




>     def test_uniqueness(
>         self,
>         split=0.5
>     )





##### Method `test_uniqueness_mmh3` {#id}




>     def test_uniqueness_mmh3(
>         self,
>         split=0.5
>     )





### Class `TestSampleIdentification` {#id}




>     class TestSampleIdentification(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `sample_id` {#id}







##### Variable `samples` {#id}










#### Methods



##### Method `test_merge_sample_identification` {#id}




>     def test_merge_sample_identification(
>         self
>     )





### Class `TestSamplesExplosion` {#id}




>     class TestSamplesExplosion(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `samples` {#id}










#### Methods



##### Method `test_sample_explode` {#id}




>     def test_sample_explode(
>         self
>     )







# Module `brevettiai.tests.test_data_image` {#id}








## Classes



### Class `TestImageAugmentation` {#id}




>     class TestImageAugmentation(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)







#### Methods



##### Method `test_augmentation_config` {#id}




>     def test_augmentation_config(
>         self
>     )





##### Method `test_image_deformation` {#id}




>     def test_image_deformation(
>         self
>     )





##### Method `test_image_filtering` {#id}




>     def test_image_filtering(
>         self
>     )





##### Method `test_image_noise` {#id}




>     def test_image_noise(
>         self
>     )





##### Method `test_image_transformation` {#id}




>     def test_image_transformation(
>         self
>     )





### Class `TestImagePipeline` {#id}




>     class TestImagePipeline(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)







#### Methods



##### Method `test_image_pipeline_from_config` {#id}




>     def test_image_pipeline_from_config(
>         self
>     )





##### Method `test_image_pipeline_get_schema` {#id}




>     def test_image_pipeline_get_schema(
>         self
>     )







# Module `brevettiai.tests.test_data_manipulation` {#id}








## Classes



### Class `TestDataManipulation` {#id}




>     class TestDataManipulation(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)







#### Methods



##### Method `test_tile2d` {#id}




>     def test_tile2d(
>         self
>     )







# Module `brevettiai.tests.test_image_loader` {#id}








## Classes



### Class `TestCropResizeProcessor` {#id}




>     class TestCropResizeProcessor(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `test_image_path` {#id}










#### Methods



##### Method `test_loader_affine_transform` {#id}




>     def test_loader_affine_transform(
>         self
>     )





### Class `TestImagePipelineToImageLoaderConversion` {#id}




>     class TestImagePipelineToImageLoaderConversion(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `test_image_path` {#id}










#### Methods



##### Method `test_ensure_default_settings` {#id}




>     def test_ensure_default_settings(
>         self
>     )





##### Method `test_ensure_roi` {#id}




>     def test_ensure_roi(
>         self
>     )





##### Method `test_ensure_roi_and_target_size` {#id}




>     def test_ensure_roi_and_target_size(
>         self
>     )





##### Method `test_ensure_target_size` {#id}




>     def test_ensure_target_size(
>         self
>     )







# Module `brevettiai.tests.test_model_metadata` {#id}








## Classes



### Class `ParsingException` {#id}




>     class ParsingException(
>         *args,
>         **kwargs
>     )


Common base class for all non-exit exceptions.



#### Ancestors (in MRO)

* [builtins.Exception](#builtins.Exception)
* [builtins.BaseException](#builtins.BaseException)







### Class `TestModelMetadata` {#id}




>     class TestModelMetadata(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `image_segmentation_metadata` {#id}







##### Variable `metadata` {#id}










#### Methods



##### Method `test_ensure_all_examples_are_valid_with_modelmetadata` {#id}




>     def test_ensure_all_examples_are_valid_with_modelmetadata(
>         self
>     )





##### Method `test_ensure_examples_are_valid_with_imagesegmentationmodelmetadata` {#id}




>     def test_ensure_examples_are_valid_with_imagesegmentationmodelmetadata(
>         self
>     )







# Module `brevettiai.tests.test_pivot` {#id}








## Classes



### Class `TestPivotIntegration` {#id}




>     class TestPivotIntegration(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `samples` {#id}










#### Methods



##### Method `test_data` {#id}




>     def test_data(
>         self
>     )





##### Method `test_fields` {#id}




>     def test_fields(
>         self
>     )







# Module `brevettiai.tests.test_platform_job` {#id}








## Classes



### Class `TestPlatformJob` {#id}




>     class TestPlatformJob(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)







#### Methods



##### Method `test_extra_arguments_on_job` {#id}




>     def test_extra_arguments_on_job(
>         self
>     )





##### Method `test_job_create_schema` {#id}




>     def test_job_create_schema(
>         self
>     )





##### Method `test_job_lifecycle` {#id}




>     def test_job_lifecycle(
>         self
>     )







# Module `brevettiai.tests.test_polygon_extraction` {#id}







## Functions



### Function `plot_mask_and_contours` {#id}




>     def plot_mask_and_contours(
>         mask,
>         contours
>     )






## Classes



### Class `TestPolygonExtraction` {#id}




>     class TestPolygonExtraction(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)







#### Methods



##### Method `test_polygon_validity` {#id}




>     def test_polygon_validity(
>         self
>     )





##### Method `test_polygon_validity_random_field` {#id}




>     def test_polygon_validity_random_field(
>         self
>     )







# Module `brevettiai.tests.test_schema` {#id}








## Classes



### Class `TestSchema` {#id}




>     class TestSchema(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)







#### Methods



##### Method `test_build_schema` {#id}




>     def test_build_schema(
>         self
>     )







# Module `brevettiai.tests.test_tags` {#id}








## Classes



### Class `TestTags` {#id}




>     class TestTags(
>         methodName='runTest'
>     )


A class whose instances are single test cases.

By default, the test code itself should be placed in a method named
'runTest'.

If the fixture may be used for many test cases, create as
many test methods as are needed. When instantiating such a TestCase
subclass, specify in the constructor arguments the name of the test method
that the instance is to execute.

Test authors should subclass TestCase for their own tests. Construction
and deconstruction of the test's environment ('fixture') can be
implemented by overriding the 'setUp' and 'tearDown' methods respectively.

If it is necessary to override the __init__ method, the base class
__init__ method must always be called. It is important that subclasses
should not change the signature of their __init__ method, since instances
of the classes are instantiated automatically by parts of the framework
in order to be run.

When subclassing TestCase, you can set these attributes:
* failureException: determines which exception will be raised when
    the instance's assertion methods fail; test methods raising this
    exception will be deemed to have 'failed' rather than 'errored'.
* longMessage: determines whether long messages (including repr of
    objects used in assert methods) will be printed on failure in *addition*
    to any explicit message passed.
* maxDiff: sets the maximum length of a diff in failure messages
    by assert methods using difflib. It is looked up as an instance
    attribute so can be configured by individual tests if required.

Create an instance of the class that will use the named test
method when executed. Raises a ValueError if the instance does
not have a method with the specified name.



#### Ancestors (in MRO)

* [unittest.case.TestCase](#unittest.case.TestCase)




#### Class variables



##### Variable `tag1` {#id}







##### Variable `tag2` {#id}







##### Variable `tag3` {#id}







##### Variable `tag4` {#id}







##### Variable `tag5` {#id}







##### Variable `tree` {#id}










#### Methods



##### Method `test_find_named_tag` {#id}




>     def test_find_named_tag(
>         self
>     )





##### Method `test_find_path` {#id}




>     def test_find_path(
>         self
>     )







# Module `brevettiai.utils` {#id}





## Sub-modules

* [brevettiai.utils.argparse_utils](#brevettiai.utils.argparse_utils)
* [brevettiai.utils.dict_utils](#brevettiai.utils.dict_utils)
* [brevettiai.utils.model_version](#brevettiai.utils.model_version)
* [brevettiai.utils.module](#brevettiai.utils.module)
* [brevettiai.utils.numpy_json_encoder](#brevettiai.utils.numpy_json_encoder)
* [brevettiai.utils.pandas_utils](#brevettiai.utils.pandas_utils)
* [brevettiai.utils.polygon_utils](#brevettiai.utils.polygon_utils)
* [brevettiai.utils.profiling](#brevettiai.utils.profiling)
* [brevettiai.utils.singleton](#brevettiai.utils.singleton)
* [brevettiai.utils.tag_utils](#brevettiai.utils.tag_utils)
* [brevettiai.utils.tf_serving_request](#brevettiai.utils.tf_serving_request)
* [brevettiai.utils.validate_args](#brevettiai.utils.validate_args)







# Module `brevettiai.utils.argparse_utils` {#id}







## Functions



### Function `overload_dict_from_args` {#id}




>     def overload_dict_from_args(
>         args,
>         target,
>         errors_ok=True
>     )





### Function `parse_args_from_dict` {#id}




>     def parse_args_from_dict(
>         args,
>         target
>     )








# Module `brevettiai.utils.dict_utils` {#id}







## Functions



### Function `dict_merger` {#id}




>     def dict_merger(
>         source,
>         target
>     )


Merge two dicts of dicts
:param source:
:param target:
:return:


### Function `in_dicts` {#id}




>     def in_dicts(
>         d,
>         uri
>     )


Check if path of keys in dict of dicts
:param d: dict of dicts
:param uri: list of keys
:return:





# Module `brevettiai.utils.model_version` {#id}







## Functions



### Function `check_model_version` {#id}




>     def check_model_version(
>         fname=None,
>         version=None,
>         fileobj=None,
>         blength=4
>     )


Check model version is correct
:param fname: path to model archive, or filelike obj
:param fileobj: BytesIO if model in memory
:param blength: bytelength of model version (4)
:return: Version number of file if correct, False if not correct and None if not a known file type


### Function `get_model_version` {#id}




>     def get_model_version(
>         obj,
>         blength=4
>     )


Calculate version of model archive
:param obj: fname path to model archive or BufferedIOBase
:param blength: bytelength of model version (4)
:return:


### Function `package_saved_model` {#id}




>     def package_saved_model(
>         saved_model_dir,
>         name='saved_model',
>         output=None,
>         model_meta=None
>     )


Utility function to package saved model to archive
:param saved_model_dir: path to saved model directory
:param name: name of model
:param output: forced output path use given name in place of name with model version
:return: archive path


### Function `saved_model_model_meta_filename` {#id}




>     def saved_model_model_meta_filename(
>         saved_model_dir
>     )








# Module `brevettiai.utils.module` {#id}







## Functions



### Function `get_parameter_type` {#id}




>     def get_parameter_type(
>         parameter
>     )






## Classes



### Class `Module` {#id}




>     class Module


Base class for serializable modules




#### Descendants

* [brevettiai.interfaces.vue_schema_utils.VueSettingsModule](#brevettiai.interfaces.vue_schema_utils.VueSettingsModule)
* [brevettiai.io.tf_recorder.TfRecorder](#brevettiai.io.tf_recorder.TfRecorder)





#### Static methods



##### `Method from_config` {#id}




>     def from_config(
>         config
>     )





##### `Method validator` {#id}




>     def validator(
>         x
>     )






#### Methods



##### Method `copy` {#id}




>     def copy(
>         self
>     )





##### Method `get_config` {#id}




>     def get_config(
>         self
>     )







# Module `brevettiai.utils.numpy_json_encoder` {#id}








## Classes



### Class `NumpyEncoder` {#id}




>     class NumpyEncoder(
>         *,
>         skipkeys=False,
>         ensure_ascii=True,
>         check_circular=True,
>         allow_nan=True,
>         sort_keys=False,
>         indent=None,
>         separators=None,
>         default=None
>     )


Extensible JSON <http://json.org> encoder for Python data structures.

Supports the following objects and types by default:

+-------------------+---------------+
| Python            | JSON          |
+===================+===============+
| dict              | object        |
+-------------------+---------------+
| list, tuple       | array         |
+-------------------+---------------+
| str               | string        |
+-------------------+---------------+
| int, float        | number        |
+-------------------+---------------+
| True              | true          |
+-------------------+---------------+
| False             | false         |
+-------------------+---------------+
| None              | null          |
+-------------------+---------------+

To extend this to recognize other objects, subclass and implement a
<code>.default()</code> method with another method that returns a serializable
object for <code>o</code> if possible, otherwise it should call the superclass
implementation (to raise <code>TypeError</code>).

Constructor for JSONEncoder, with sensible defaults.

If skipkeys is false, then it is a TypeError to attempt
encoding of keys that are not str, int, float or None.  If
skipkeys is True, such items are simply skipped.

If ensure_ascii is true, the output is guaranteed to be str
objects with all incoming non-ASCII characters escaped.  If
ensure_ascii is false, the output can contain non-ASCII characters.

If check_circular is true, then lists, dicts, and custom encoded
objects will be checked for circular references during encoding to
prevent an infinite recursion (which would cause an OverflowError).
Otherwise, no such check takes place.

If allow_nan is true, then NaN, Infinity, and -Infinity will be
encoded as such.  This behavior is not JSON specification compliant,
but is consistent with most JavaScript based encoders and decoders.
Otherwise, it will be a ValueError to encode such floats.

If sort_keys is true, then the output of dictionaries will be
sorted by key; this is useful for regression tests to ensure
that JSON serializations can be compared on a day-to-day basis.

If indent is a non-negative integer, then JSON array
elements and object members will be pretty-printed with that
indent level.  An indent level of 0 will only insert newlines.
None is the most compact representation.

If specified, separators should be an (item_separator, key_separator)
tuple.  The default is (', ', ': ') if *indent* is <code>None</code> and
(',', ': ') otherwise.  To get the most compact JSON representation,
you should specify (',', ':') to eliminate whitespace.

If specified, default is a function that gets called for objects
that can't otherwise be serialized.  It should return a JSON encodable
version of the object or raise a <code>TypeError</code>.



#### Ancestors (in MRO)

* [json.encoder.JSONEncoder](#json.encoder.JSONEncoder)







#### Methods



##### Method `default` {#id}




>     def default(
>         self,
>         obj
>     )


Implement this method in a subclass such that it returns
a serializable object for <code>o</code>, or calls the base implementation
(to raise a <code>TypeError</code>).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)




# Module `brevettiai.utils.pandas_utils` {#id}







## Functions



### Function `explode` {#id}




>     def explode(
>         df,
>         on=None,
>         fillna='N/A',
>         duplicate_id='id',
>         keep_empty=True
>     )


Explode all explodable columns in dataframe, see: pd.DataFrame.explode

Count unique items by grouping on all columns, counting each group size, then dropping duplicate ids
df.groupby(df.columns.tolist()).size().reset_index(name="count").drop_duplicates("id")["count"].sum()

:param df:
:param on: explode on columns
:param fillna: fill NA's with the following value
:param duplicate_id: column on return df to set group duplication id or None to avoid grouping
:param keep_empty: keep empty lists as NAN rows
:return: see: pd.DataFrame.explode





# Module `brevettiai.utils.polygon_utils` {#id}







## Functions



### Function `cv2_contour_to_shapely` {#id}




>     def cv2_contour_to_shapely(
>         contour,
>         hole=False,
>         resolution=2
>     )





### Function `simplify_polygon` {#id}




>     def simplify_polygon(
>         polygon,
>         min_=0.2,
>         max_=3,
>         alpha=0.005,
>         preserve_topology=True
>     )








# Module `brevettiai.utils.profiling` {#id}







## Functions



### Function `profile_graph` {#id}




>     def profile_graph(
>         graph
>     )





### Function `profile_graph_def` {#id}




>     def profile_graph_def(
>         graph_def
>     )





### Function `profile_keras_model` {#id}




>     def profile_keras_model(
>         model,
>         batch_size=1,
>         shape_override=None
>     )


shape override takes precedence over batch_size
:param model:
:param batch_size:
:param shape_override:
:return:





# Module `brevettiai.utils.singleton` {#id}








## Classes



### Class `Singleton` {#id}




>     class Singleton(
>         cls
>     )











#### Methods



##### Method `Instance` {#id}




>     def Instance(
>         self
>     )







# Module `brevettiai.utils.tag_utils` {#id}







## Functions



### Function `find` {#id}




>     def find(
>         tree,
>         key,
>         value
>     )





### Function `find_path` {#id}




>     def find_path(
>         tree,
>         key,
>         value,
>         path=()
>     )








# Module `brevettiai.utils.tf_serving_request` {#id}







## Functions



### Function `parse_args` {#id}




>     def parse_args()





### Function `tf_serving_request` {#id}




>     def tf_serving_request(
>         ip,
>         port,
>         version,
>         model,
>         images,
>         repeat=1,
>         show=False,
>         token=None,
>         resize='',
>         **kwargs
>     )








# Module `brevettiai.utils.validate_args` {#id}








## Classes



### Class `ValidateArgs` {#id}




>     class ValidateArgs(
>         validator,
>         throw=True
>     )


Decorator for validating parameters of function