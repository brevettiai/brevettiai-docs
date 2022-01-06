# Model training
Once your model and your virtual environment is correctly set up, you may start a model training using a command prompt.

To start a model training, you first need to provide the prompt with an API key and job ID for the model you wish to use. These can be extracted from the Brevetti AI platform.

### Job ID
To extract the job ID, navigate to your model in the Brevetti AI platform. The Job ID is found in your browser address field, being the code listed after the last '/'.
(F.ex. "https://platform.brevetti.ai/models/**806fd8f4-6675-45d0-a143-5807a7a50741**")

### API key
The API key is also extracted from your model page. Select "Options --> Show API Key" in the right side of the screen, and a pop-up with the API key will be shown (if you have access rights to it).

## Initiate the model training
The model training may finally be started either through a python IDE or in a command prompt. In case of the image-segmentation library, start the training with the following command:
```
python -m image_segmentation.train --model_id yy --api_key xx

```
where yy and xx are the respective keys.

If everything goes well, you will eventually start seeing epochs and evaluation metrics being displayed in the terminal. Once the training is done, you can access the training metrics on the Brevetti AI platform. 

## Troubleshooting common problems

### ...no known parent package
When running the above command, you may encounter the following error message:

```
importerror: attempted relative import with no known parent package.
```

This problem can be circumvented in the following way. In Pycharm, navigate to the folder containing your training scripts (in the example's case, image_segmentation). First, select the main folder and mark the directory as "Set as sources root". After this, open up the run/debug configuration terminal, choose "module name" and write "image_segmentation.run_job" in the name as well as module name bar. Apply and run it again, and the problem should be fixed.

If using another terminal than PyCharm, similar applications to the run/debug configuration method may exist.

### AttributeError: 'Series' object has no attribute '_is_builtin_func'

This error may be caused by an outdated package. For example, the tqdm package once caused this error. The problem was simply fixed by updating the tqdm package to its latest version and rerunning the script.
