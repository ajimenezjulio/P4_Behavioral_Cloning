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
 File                         | Description                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                   | Implements generators, model architecture and runs the training pipeline.          |
| `model.h5`                   | Model weights.                                                                     |
| `dataset_utilities.py`       | Load, filter and extend the dataset.                                               |
| `image_preprocessing.py`     | All image preprocessing methods are including here.                                |
| `video.py`     			           | Convert sequence of images into a video.                                           |
| `drive.py`                   | Communicates with the driving simulator app providing model predictions based on real-time data simulator app. |
| `Behavioral_Cloning.ipynb.py`| Jupyter notebook for a better visualization about the steps involved in the project.|

### 1. Examine the dataset

The dataset presents steering values from `-1 to +1` however there's a challenge, taking the histogram we can appreciate that most of the values are around 0, which unbalances the dataset.

![Histogram_Before_Filtering](https://github.com/ajimenezjulio/P4_Behavioral_Cloning/blob/master/Markdown_Images/histo_1.png)


### 2. Filter and extend the dataset

In order to balance the dataset, this approach first remove all those values very close to 0 and then extend the dataset with extreme values, resulting in still a normal distribution but with more dispersed values. After this step we ended up with a dataset of `17142` samples.

![Histogram_After_Filtering](https://github.com/ajimenezjulio/P4_Behavioral_Cloning/blob/master/Markdown_Images/histo_2.png)


### 3. Image perturbation

The following steps were performed to augment, distort and perturb the dataset.

1. **Random image picking:** At every row we randomly pick an image from either the left, center or right camera.
2. **Random translation:** We add a random translation to the image and compensate the steering angle accordingly. 
3. **Random brightness:** The brightness of the image is randomly increased, the brightness is added in a HSV colorspace so the impact will be less aggressive and then transformed back to RGB.
4. **Random shadow:** A random shadow is added to the image in an attempt to improve generalization.

![Image_Perturbation](https://github.com/ajimenezjulio/P4_Behavioral_Cloning/blob/master/Markdown_Images/perturb.png)

### 4. Image preprocessing 
Before feeding the network with an image, some steps need to be performed:

1. **Cropping:** The sky  and front part of the vehicle are distractors that add no value to the network.
2. **Resize:** A `64 x 64` image size was picked for this project, the squared size promises to highlight in a better way the features of the road like the edge lines and curves (more like a bird eye perspective).
3. **HSV transform:** HSV is more illumination invariant than RGB, so it was picked for training the network.

![Image_Preprocessing](https://github.com/ajimenezjulio/P4_Behavioral_Cloning/blob/master/Markdown_Images/preprocess.png)

### 4. Model

The model used for this project is a modified version of the [Nvidia paper](https://arxiv.org/abs/1604.07316), it consists of 1 normalization, 3 convolutional and 3 fully connected layers in the following way.

![CNN](https://github.com/ajimenezjulio/P4_Behavioral_Cloning/blob/master/Markdown_Images/behavioral_cloning.pdf)


### 5. Results

The model performs very well for the first track and fairly good for the second one, however there are multiple challenges involving landscape colors and sharp curves where the model can't react fast enough or accurate enough and the car will lose track of the road.
