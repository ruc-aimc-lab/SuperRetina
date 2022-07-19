# Tutorial code

## Inference of SuperRetina to realize the registration of two retinal images.

Note that before running [tutorial-inference.ipynb](./tutorial-inference.ipynb), make sure 
you have prepared the config file in `config` file. Here we give a demo config file [test.yaml](../config/test.yaml).

+ Note in notebooks we use `os.chdir("..")` to set the working directory to the root director.
+ `model_save_path` is the path of the saved model of SuperRetina.
+ You can adjust the parameters `model_image_width`, `model_image_height`, `nms_size`, `nms_thresh` and `knn_thresh` to get different results of registration, which `model_image_width` and `model_image_height` is used to resize the input  of SuperRetina, `knn_thresh` is the threshold of the ratio test.


## Inference of SuperRetina to evaluate the registration performance on FIRE.
Similarly, [eval-registration-on-FIRE](./eval-registration-on-FIRE.ipynb) also need the config file [test.yaml](../config/test.yaml). 

+ Note that you should download FIRE dataset before running the evaluation. Then indicate the path of FIRE in the tutorial file. You can download the dataset from [here](https://projects.ics.forth.gr/cvrl/fire/).
+ We provide an interface class `Predictor` in [predictor.py](../predictor.py) to help to evaluate SuperRetina. The usage of this class can be found in the notebook.