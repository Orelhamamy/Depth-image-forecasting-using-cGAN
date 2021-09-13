# Depth image forecasting
By Orel Hamamy(https://github.com/Orelhamamy) and [Or Tslil](https://github.com/ortslil64). This project is a final 
BSc project.
 
This project includes:
1) ROS package generates a sequence of depth images, with two Gazebo worlds. 
2) Python scripts for generating, training, and evaluating cGAN models(2D&3D-convolution).
3) Implementation of a trained model within the simulated environment.

## Example
Deploying of forecasting depth images model within Gazebo simulation:

[![Watch the video](https://img.youtube.com/vi/QhmAMWtSH_I/hqdefault.jpg)](https://www.youtube.com/watch?v=QhmAMWtSH_I)

## Setup

### Prerequisites and Dependencies
- Ubuntu
- Python 2.*, 3.*
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

The following python packages are required:
- TensorFlow 2.*
- Numpy
- OpenCv2
- matplotlib 
- scipy
- cv_bridge
- rospy
- getkey

This project was tested within Ubuntu 18.04 and ROS melodic.

### Getting started
1. Clone this repository to your catkin workspace:
```bash
cd $HOME/catkin_ws/src
git clone https://github.com/Orelhamamy/Depth-image-forecasting-using-cGAN.git
```
2. Build:
```bash
catkin_make
```
3. (recommended) source the packge to .bashrc:
```bash 
echo "source $(pwd)/devel/setup.bash" >> $HOME/.bashrc
```
## Generating a dataset
There are two options, using generated benchmark (include in this repo) either to generate your own. If you are using the generated benchmark, continue to the next stage.

To generate your own follow those steps:
1. Launch the Gazebo simulation:
```bash
roslaunch depth_image_forecasting gazebo_train_world.launch 
```
2. Run the teleop keyboard to drive the robot, alternative use `rqt`:
```bash
rosrun depth_image_forecasting teleop_twist_keyboard_keys.py 
```

Use the arrows to move around, q - for increase speeds, a - decrease speeds and, s - stop the robot.

3. Run the capturing images node:
```bash
rosrun depth_image_forecasting data_set_creator.py -usage
```
The usage arg can be train or test. **Note** this node will erase all previous data from train or test, where generating one dataset enforces correlating the second.

4. Choose (click) the terminal where the teleop running, and drive around the simulated world while the robot capturing depth images.
## Train
(optional) Download a pre-trained model and skip this stage, you can download from [here](https://drive.google.com/drive/folders/1Z-qdWwg0yoYM70rIichATgCwB4QbSh_T?usp=sharing) then save it in `/Depth-image-forecasting-using-cGAN/models`.
It's recommended to use Spyder or your favorite python IDE, for training a model. With IDE hyperparameters tunning is easier.
For training within a terminal run: 
```bash
cd $HOME/catkin_ws/src/Depth-image-forecasting-using-cGAN/models
python3 model_train.py 
```
## Test

### Graphs generation
For plotting the loss functions values use MATLAB (we use R2018a), to run `plot_models_loss_functions.m` script.

### Visual results

Run the states and prediction script: 
```bash
python3 states_and_predictions.py -model_name -states
```
The states arg is an integer (recommended to be [10,20]), the default is 10.
exmple: 
![demo](https://github.com/Orelhamamy/Depth-image-forecasting-using-cGAN/blob/master/images/states_and_predictions.png?raw=true "States and predictions")

Rows are arranged from top-down: Ground-truth, predicted k+1 state, predicted k+2 state using a previous predicted state (k+1). 
### Features visualization

Run the features visualization script: 
```bash
python3 feature_visualization.py -model_name -index
```
The index argument is the data set index of the model's input sequence, either change the default input index within the script.
Running this script within Spyder will simplify the execution. 
This script will save the hidden layers output in: `/models/$(model_name)/features/$(model_name)_$(layer_number)`. The script will generate a `.png` and `.eps` files.

Feature visualization, first hidden layer for 3D model:
![demo](https://github.com/Orelhamamy/Depth-image-forecasting-using-cGAN/blob/master/images/SM-3D_conv_feature-1.png?raw=true "First hidden layer")
## 3D convulation model

The 3D convolution is an API within python, the script is at: `/models/three_d_conv_model`. You can choose to load an existing model or initialize one. Either way, it's recommended to train, test, and modify the 3D models within an IDE. Example to initialize, training, and validation the 3D model: 
```python

from three_d_conv_model import Three_d_conv_model

model_name = 'SM-3D_conv'
root_path = os.path.abspath(__file__ + "/../..")
test_set_path = root_path + "/data_set/test/"
train_set_path = root_path + "/data_set/train/"

# --- loading existe model --- # 
model = Three_d_conv_model(model_name,
                        load_model=True)
 
# --- initialize model --- # 
model = Three_d_conv_model(model_name, train_set_path, 
                           OBSERVE_SIZE = 5, load_model = False)

# --- print scheme of the generator and discriminator --- # 
model.print_model()

# --- train the model --- #
model.fit(150, model_name, disc_reff=False)

# --- train the model with a reffernce discriminator --- # 
model.fit(150, model_name, disc_reff=True)

# --- validate the model --- #
model.model_validation(0,350,test_path=test_set_path) 
``` 

## Deployed the model

This section introduces how to forecast depth images from a running simulation. This deployed works only with a Recursive model or a 3D model, a model with a prediction gap>0 can't be executed.

1. Launch the simulation and the robot:
```bash
roslaunch depth_image_forecasting gazebo_test_world.launch
```

2. Run a publisher of depth image sequences:
```bash
rosrun depth_image_forecasting image_sequence_publisher.py 
```

3. Run the forecasting model: 
```bash
rosrun depth_image_forecasting live_forecasting.py --model_name SM-3D_conv
```

The arg `--model_name` is the name of a trained model from `/models` directory. A pre-trained model can be found [here](https://drive.google.com/drive/folders/1Z-qdWwg0yoYM70rIichATgCwB4QbSh_T?usp=sharing).


