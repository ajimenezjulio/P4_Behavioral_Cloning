## Behavioral Cloning
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project we aim to predict the steering angle a driver must perform based on images collected from the road using a deep neural network trained on Keras. Along with the simulator app we can simulate a self driving behavior of a car.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Dependencies

- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](http://tensorflow.org)
- [Pandas](http://pandas.pydata.org/)
- [OpenCV](http://opencv.org/)
- [Matplotlib](http://matplotlib.org/)
- [Jupyter](http://jupyter.org/)

### Project structure
 File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                   | Implements generators, model architecture and runs the training pipeline.             |
| `model.h5`                   | Model weights.                                                                     |
| `dataset_utilities.py`       | Load, filter and extend the dataset.                      
|
| `image_preprocessing.py`     | All image preprocessing methods are including here.                      
|
| `video.py`     			   | Convert sequence of images into a video.                      
|
| `drive.py`                   | Communicates with the driving simulator app providing model predictions based on real-time data simulator app. |
| `Behavioral_Cloning.ipynb.py`| Jupyter notebook for a better visualization about the steps involved in the project.                      
|

### 1. Examine the dataset

The dataset presents steering values from `-1 to +1` however there's a challenge, taking the histogram we can appreciate that most of the values are around 0, which unbalances the dataset.

![Data_Representation](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Representation.png)


### 2. Filter and extend the dataset

In order to balance the dataset, this approach first remove all those values very close to 0 and then extend the dataset with extreme values, resulting in still a normal distribution but with more dispersed values. After this step we ended up with a dataset of `17142` samples.

![Data_Distribution_Before_Balancing](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Distribution_Before_Balancing.png)


### 3. Image perturbation

The following steps were performed to augment, distort and perturb the dataset.

1. **Random image picking:** At every row we randomly pick an image from either the left, center or right camera.
2. **Random translation:** We add a random translation to the image and compensate the steering angle accordingly. 
3. **Random brightness:** The brightness of the image is randomly increased, the brightness is added in a HSV colorspace so the impact will be less aggressive and then transformed back to RGB.
4. **Random shadow:** A random shadow is added to the image in an attempt to improve generalization.

![Data_Distribution_Before_Balancing](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Distribution_After_Balancing.png)

### 4. Image preprocessing 
Before feeding the network with an image, some steps need to be performed:

1. **Cropping:** The sky  and front part of the vehicle are distractors that add no value to the network.
2. **Resize:** A `64 x 64` image size was picked for this project, the squared size promises to highlight in a better way the features of the road like the edge lines and curves (more like a bird eye perspective).
3. **HSV transform:** HSV is more illumination invariant than RGB, so it was picked for training the network.

![Data_Distribution_Before_Balancing](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Data_Distribution_After_Balancing.png)

### 4. Model

The model used for this project is a modified version of the [Nvidia paper](https://arxiv.org/abs/1604.07316), it consists of 3 convolutional layers and 3 fully connected in the following way.

![Lenet](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Lenet.png)


### 5. Results

The model performs very well for the first track and fairly good for the second one, however there are multiple challenges involving landscape colors and sharp curves where the model can't react fast enough or accurate enough and the car will lose track of the road.

| Epoch | Training set | Validation set | Test set |
|:-----:|:------------:|:--------------:|:--------:|
| 1 | 0.031814 | 0.007483 | 0.007047 |
| 5 | 0.256081 | 0.265306 | 0.283769 |
| 10 | 0.559570 | 0.537642 | 0.539667 |
| 50 | 0.962407 | 0.953515 | 0.948614 |
| **200** | **0.984674** | **0.976417** | **0.975693** |

![Accuracy_Training](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Results/3conv-2fc%203-16-32-64-1024-84-43__0.0005_128_Final.jpeg)

After 200 epochs the model could get above 97% of accuracy for all datasets which makes it a fairly decent approach.

### 6. Feature maps

In order to understand the weights the model is using for the classification task in a better way, it is a good idea to visualize them, so we can have an idea of the abstract patterns the model is selecting. Below samples of the feature maps are displayed.

![Feature_Maps](https://github.com/ajimenezjulio/P3_Traffic_Sign_Recognition/blob/master/Markdown/Feature_Maps.png)