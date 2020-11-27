import tensorflow as tf

# Platform features useful for accessing the job parameters
from brevettiai.platform import Job, get_image_samples
from brevettiai.interfaces import vue_schema_utils as vue
from brevettiai.utils.model_version import package_saved_model # packages model for upload

# Data science tools for training an image classifier model
from brevettiai.data.image import ImagePipeline
from brevettiai.data.data_generator import DataGenerator, OneHotEncoder


def build_image_classifier(classes, image_shape):
    # Build model
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=image_shape, include_top=False, weights="imagenet"
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(len(classes), activation='softmax', name="---".join(classes))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class TrainingJob(Job):
    class Settings(vue.VueSettingsModule):
        def __init__(self, image_pipeline: ImagePipeline, epochs: int = 10):
            self.image_pipeline = image_pipeline # object to read image and resize to specified shape
            self.epochs = epochs # number of training epochs
    settings: Settings

    def train(self):
        # Get the samples and the classes from the job datasets 
        samples = get_image_samples(self.datasets)
        classes = samples.folder.unique()

        # Setup up data generator to loop through the samples
        data_generator = DataGenerator(samples, batch_size=4, output_structure=("img", "onehot"), shuffle=True, repeat=True)\
            .map([self.settings.image_pipeline, OneHotEncoder(classes=classes)])

        # Construct a keras image classifier model and train it using the data generator
        model = build_image_classifier(classes, self.settings.image_pipeline.output_shape)
        model.fit(data_generator.get_dataset(), epochs=self.settings.epochs, steps_per_epoch=len(data_generator))

        # Save model and package it along with meta data
        model.save("saved_model", overwrite=True, include_optimizer=False)
        return package_saved_model("saved_model", model_meta={"important_model_meta": 42})


if __name__ == "__main__":
    import sys, json
    # Run the script with argument --serialize_schema to get a platform settings schema written
    if "--serialize_schema" in sys.argv:
        schema = TrainingJob.Settings.get_schema().schema
        json.dump(schema, open("settings_schema.json", "w"))
    else:
        # Using sagemaker hyperparameters the TrainingJob instantiates
        # with settings and dataset access configured by the platform
        job = TrainingJob.init()
        # The train function optimizes the model and returns a path to the model artifact
        output_path = job.train()
        # The job uploads the model artifact, and closes 
        job.complete_job(output_path)
