# Clone-Driving-Behavior-SDC-Nanodegree

Deploy of NVIDIA's convnet architecture to clone driving behaviour in a game based environment created by Udacity. The model was trained to output a steering angle to an autonomous vehicle in order to drive on a virtual tracks.

<img src='https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/nvidia_sdc.png' width = 600 height=400 class='center'>


The goals / steps of this project are the following:

1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
5. Summarize the results with a written report

Rubric Points

Using the simulator, 3 laps were recorded with deffirent objetives. The first lap was recorded normally, the second slower but trying to show perfect behavior and the final using a lot of recoveries so the neural network could respond to situations were the car is about to be out of the road.

<img src='https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/arturo_bc.png' width = 600 height=400 src='https://www.youtube.com/watch?v=3MyVkS9Rr9s'>

The architecture for the neural network was similar to the one implemented by NVIDIA, which face a similar problem receiving 3 different images from center, left and right and designed a successful convolutional neural network to output a steering wheel angle:

<img src='https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/nvidia_arch.png' width = 600 height=400 class='center'>


The unique variation is that the ReLu activation function for the hidden layers is replaced by Exponential Linear Unit ELU, this modification was donde for the following reassons:

    ELU becomes smooth slowly until its output equal to -α  value whereas RELU sharply smoothes.
    Unlike to ReLU, ELU can produce negative outputs.
    Using derivatives from the negative side avoids 'death neurons' 

https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/activations.png

A generator function was used in the input of the NN in order to get some extra data generated on the fly, this is a better practice than create it and store it in the hard disk since it could represent a very important size.

The output model can be seen applied in the simulator in the following video : 

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

    model.py containing the script to create and train the model
    drive.py for driving the car in autonomous mode
    model.h5 containing a trained convolution neural network
    writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.json

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 109-143)

<img src='https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/model_summary.png' width = 600 height=400 class='center'>


The model includes ELU layers to introduce nonlinearity and good performance dealing with negative values (code line 109), and the data is normalized in the model using a normalize function (code line 39).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 112).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 172). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Data augmentation, increasing the number of training samples by performing some variations to the original data and create synthetic new samples:


<img src='https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/aug.png' width = 600 height=400 class='center'>


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 185).

https://raw.githubusercontent.com/apolanco3225/Clone-Driving-Behavior-SDC-Nanodegree/master/CarND-Behavioral-Cloning-P3-master/images/adam.png

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road applying a compensation en each of the sides that would drive the car back to the center of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement a model that was succesfully implemented before by NVIDIA's end to end aproach for steering a car using deep learning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model reducing the number of epoches where there was detected a reduction in the performance in the model and adding dropout layers in the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases the best solution to take besides to improving the model is to improve your data, adding extra examples of recovery movements showed to be the best strategy to keep the car driving in the center of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 104) consisted of a convolution neural network with the following layers and layer sizes:


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


I then recorded the vehicle recovering from the left side and right sides of the road back to center.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would multiply my data by 3. For example, here is an image that has then been flipped:


After the collection process, I then preprocessed this data by using a good number of functions like normalization, randomly change the brightness, flip images, resize and crop.

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model.json The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
