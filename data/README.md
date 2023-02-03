
# Data 

## Datasets
### Training
+ Lab
+ Auxiliary
### Test set for image registration
+ FIRE: [https://projects.ics.forth.gr/cvrl/fire/](https://projects.ics.forth.gr/cvrl/fire/)
### Test sets for identity verification
+ VARIA: [http://www.varpa.es/research/biometrics.html](http://www.varpa.es/research/biometrics.html)
+ BES
+ CLINICAL


## Data Organization

### Training data

Our semi-supervised training code reads a *labeled* training dataset (`Lab` in our ECCV paper) and an *unlabeled* training dataset (`Auxiliary`), 
and assumes each training data to be organized as follows.
```
Lab/
    Annotations/
        vistel0_left_0.txt
        vistel0_left_1.txt
        ....       
    ImageData/
        vistel0_left_0.jpg
        vistel0_left_1.jpg
        ....     
    ImageSets/
        eccv22_train.txt  
        eccv22_val.txt    
        Lab.txt  
Auxiliary/
    image1.jpg
    image2.jpg
    ...
```
+ The `Annotations` folder contains keypoint annotations per image. The folder is optional for unlabeled data. A sample annotation file is given as [samples/vistel0_left_0.txt](samples/vistel0_left_0.txt). See the [tutorial code](../notebooks/read_keypoint_labels.ipynb) that explains how the keypoint annotations shall be stored and loaded.
+ The `ImageData` folder contains all image files.
+ The `ImageSets` folder contains image-id files that specify data split. See [eccv22_train.txt](./Lab/ImageSets/eccv22_train.txt), [eccv22_val.txt](./Lab/ImageSets/eccv22_val.txt) and [lab.txt](./Lab/ImageSets/lab.txt)

### Test data
The file organizations are as follows:

```
FIRE/
    Ground Truth/
    Images/
    Masks/

VARIA/
    Images/
        R001.pgm
        R002.pgm
        ...
    pair_index.txt
```
+ Note that the annotation file of `control_points_P37_1_2.txt` in `FIRE` dataset is incorrect, so it shall be excluded from evaluation.

For identity task, `pair_index.txt` is used to indicate matching pairs of the dataset.
``` python
# each line in the index file has three colunms, means
query_image, refer_image, 0 (reject) or 1 (accept)
# for example
R180.pgm, R002.pgm, 1
R012.pgm, R002.pgm, 0
```
