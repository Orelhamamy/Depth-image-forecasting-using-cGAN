#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import cv2
import numpy as np
import rospy
import copy
import argparse
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist


#from scipy.io import savemat
#import datetime

# HEIGHT, WIDTH = 128, 128 
# depth_camera = '/depth_camera/depth/image_raw'
# model_path ='/home/lab/orel_ws/project/model_training/' + '3D_conv_5_1.3'
vel_cmd = Twist()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_topic", default= "/depth_camera/depth/image_raw/Compressed",
                        help= "The listen topic by the model.")
    parser.add_argument("--model_name", default= "3D_conv_5_1.3",
                            help="The name of the loaded model.")
    parser.add_argument("--model_path", default="/home/lab/orel_ws/project/model_training/",
                            help="The model dirctory..")
    return parser.parse_args()

'''
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
   '''     

def vel_callback(msg):
    global vel_cmd
    vel_cmd = msg


def adjust_img(img):
    # Adjust the image for generator model
    img = img*2-1
    return img[tf.newaxis,...]



def main(args):
    global vel_cmd
    model_path = args.model_path + args.model_name
    generator = tf.keras.models.load_model('{}/generator_0'.format(model_path))
    input_shape = generator.input.shape[1:-1]
    input_imgs = rospy.wait_for_message(args.input_topic, CompressedImage)
    rospy.Subscriber('/cmd_vel', Twist,vel_callback)
    vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    dividing_gap = np.ones((generator.input.shape[1],int(generator.input.shape[1]/2)))
    while not rospy.is_shutdown():
        start_time = rospy.Time.now()
        input_imgs = np.fromstring(input_imgs.data, np.float32).reshape(input_shape)
        input_imgs = adjust_img(input_imgs)
        current_frame = copy.copy(input_imgs[0,:,:,-1])
        prediction = generator(input_imgs, training = False)[0,...]
        if '3D' in args.model_name:
            prediction = prediction[...,0]
            rec_prediction = prediction[...,1]
        else:
            input_imgs = np.concatenate((input_imgs[...,1:], prediction[...,0]), axis = -1)
            rec_prediction = generator(input_imgs, training = False)[0,...,0]
        display_img = np.concatenate((current_frame, dividing_gap, prediction[...,0], dividing_gap, rec_prediction), axis=1)
        # display_img2 = cv2.normalize(display_img, None, 0, 1, cv2.NORM_MINMAX)
        display_img = (display_img + 1)/2
        # print(display_img[:,192:])
        cv2.imshow('prediction', display_img)
        cv2.waitKey(1)
        print("Computation time for {}: {} (sec)".format(args.model_name,(rospy.Time.now()-start_time).to_sec()))
        input_imgs = rospy.wait_for_message(args.input_topic, CompressedImage)
        '''
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
        '''
        
if __name__ =='__main__':
    rospy.init_node('obstacle_avoidance')
    main(parser())
