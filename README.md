# Depth image forecasting
By Orel Hamamy(https://github.com/Orelhamamy) and [Or Tslil](https://github.com/ortslil64). This project is the final 
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

The following python packges are required:
- TensorFlow 2.*
- Numpy
- OpenCv2
- matplotlib 
- scipy
- cv_bridge
- rospy
- getkey

This project tested whitin Ubuntu 18.04 and ROS melodic.

### Getting started
1. Clone this repository to your catkin workspace:
```bash
git clone https://github.com/Orelhamamy/Depth-image-forecasting-using-cGAN.git
```
2. Build:
```bash
catkin_make
```
## Generating a dataset
There are two options, using generated benchmark (include in this repo) either to generate your own. If you are using the generated benchmark, continue to the next stage.

To generate your own follow those steps:
1. Launch the Gazebo simulation:
```bash
roslaunch ... UPDATE THIS
```
2. Run the teleop keyboard to dirve the robot, alternative use `rqt`:
```bash
roslaunch ... UPDATE THIS
```
Use the arrows to move around, q - for increase speeds, a - decrease speeds and, s - stop the robot.

3. Run the capturing images node:
```bash
roslaunch ... UPDATE THIS
```
4. Choose (click) the terminal where the teleop running, and drive around the simulated world while the robot capturing depth images.
## Train

## Test

## Deployed the model
