#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import cv2
import numpy as np
import rospy
import copy

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist


#from scipy.io import savemat
#import datetime

HEIGHT, WIDTH = 128, 128 
depth_camera = '/depth_camera/depth/image_raw'
model_path ='/home/lab/orel_ws/project/model_training/' + 'cGAN_5pic_1y_train_1.9'
vel_cmd = Twist()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_topic", default= "/depth_camera/depth/image_raw/Compressed",
                        help= "The listen topic by the model.")
    parser.add_argument("--model_name", default= "cGAN_5pic_1y_train_1.9",
                            help="The name of the loaded model.")
    parser.add_argument("--model_path", default="/home/lab/orel_ws/project/model_training/",
                            help="The model dirctory..")
    return parser.parse_args()


class image_buffer():
    # Save image as input size
    def __init__(self, input_shape, bridge):
        self.img_seq = np.ones(input_shape)
        self.bridge = bridge
        
    def update_img(self, event):
        print('123\n\n\n')
        img_msg = rospy.wait_for_message(depth_camera,Image)
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        img = cv2.resize(img, (WIDTH, HEIGHT))*10/127.5-1.
        self.img_seq = np.concatenate((self.img_seq, img), axis = 2)[:,:,-5]
        print(self.img_seq.shape)
        

def vel_callback(msg):
    global vel_cmd
    vel_cmd = msg
    
def main(args):
    global vel_cmd
    generator = tf.keras.models.load_model('{}/generator_0'.format(model_path))
    bridge = CvBridge()
    input_shape = generator.input.shape[-3:]
    input_imgs = image_buffer(input_shape, bridge)
    rospy.Timer(rospy.Duration(INTERVAL), input_imgs.update_img)
    rospy.Subscriber('/cmd_vel', Twist,vel_callback)
    vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    while True:
        prediction = generator(input_imgs.img_seq[tf.newaxis,...], training = False)[0,...,0].numpy()
        cv2.imshow('prediction', prediction)
        cv2.waitKey(1)
        temp_vel = copy.copy(vel_cmd)
        if np.any(prediction[:64,:64:98]<-0.98):
            temp_vel.angular.z += 0.5
        elif np.any(prediction[:64,:64]<-0.95):
            temp_vel.angular.z -= 0.5
        if np.mean(prediction[32:96,32:64])<=-0.95:
            temp_vel.linear.x = -0.5
        vel_publisher.publish(temp_vel)
        rospy.sleep(2.5)
        vel_publisher.publish(vel_cmd)
        
if __name__ =='__main__':
    rospy.init_node('obstacle_avoidance')
    main(parser())
