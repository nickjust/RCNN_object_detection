## R-CNN for object detection and motion tracking 

### Objective:
Implementation of an coustomized motion and object detection (Computer Vision) of an arbitrary mobile object using R-CNN (Deep Learning).
The object (remote controlled car) should be recognised in individual images and in videos in order to track its movement.

### Part 1: Generation of data
- Created a custom dataset with images and bounding box labels of the remote controlled car (see folder [base_data](https://github.com/nickjust/RCNN_object_detection/tree/main/base_data)) 
- [Labelimg](https://github.com/nickjust/RCNN_object_detection/tree/main/labelimg) was used to annotate the bounding boxes and for automated saving of the coordinates in .xml format 

![Imagenesl](images_readme/labeling.png)

### Part 2: Preprocessing 
- Preprocessing of the data for the R-CNN algorithm is necessary, more precisely for training the CNN, since the CNN can not learn directly via the bounding box coordinates and expects a fixed size of the input images
- Automated creation of a separate [training dataset](https://github.com/nickjust/RCNN_object_detection/tree/main/training_data) for the CNN classification model, which contains positive and negative examples regarding the object to be classified using OpenCV library and Selective Search algorithm (see script [generate_training_data.py](https://github.com/nickjust/RCNN_object_detection/blob/main/generate_training_data.py)
- Montage of the generated images for both classes:

![Imagenesl](images_readme/auto_kein_auto_datensatz.png)


### Part 3: Training of CNN classification model
- Self-developed and trained CNN model using the libraries Tensorflow and Keras for later classification
- Hyperparametertuning 
