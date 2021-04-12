#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import time
import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import argparse
import cv2

#import datetime


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type = float, default= 0.25,
                        help= "Define the interval between cupture images.")
    parser.add_argument("--depth_camera_path", default= "/depth_camera/depth/image_raw",
                        help= "Depth camera topic within the robot.")
    parser.add_argument("--publish_topic", default= "/depth_camera/depth/image_raw/Compressed",
                            help="Publish topic for the compressed image.")
    parser.add_argument("--observe_size", default= 5,
                            help="IMPORTENT - this value have to be suited with the generative model.")
    parser.add_argument("--img_shape", default= (128,128),
                            help="IMPORTENT - this value have to be suited with the generative model.")
    return parser.parse_args()




class imgs_buffer():
    def __init__(self, topic, interval, shape):
        self.bridge = CvBridge()
        init_img = rospy.wait_for_message(topic, Image)
        self.data = cv2.resize(self.bridge.imgmsg_to_cv2(init_img, desired_encoding='passthrough'),(shape[0],shape[1]))
        self.data = np.expand_dims(self.data, -1)
        self.shape = shape
        self.rate = rospy.Rate(1/interval)
        rospy.Subscriber(topic, Image, self.img_listen, queue_size= 1)
        
    
    def img_listen(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            img = np.expand_dims(cv2.resize(img, (self.shape[0], self.shape[1])), -1)
            self.data = np.concatenate((self.data, img), axis = -1)
        except CvBridgeError:
            pass
        if (self.data.shape[-1]>self.shape[-1]):
            self.data = self.data[...,1:]
        self.rate.sleep()
    

    def compress(self):
        img = CompressedImage()
        img.header.stamp = rospy.Time.now()
        img.format = "jpeg"
        img.data = self.data.tostring()
        return img




def main(args):
    assert args.publish_topic.endswith("Compressed"), "The publish topic must end with 'Compressed'"
    publisher = rospy.Publisher(args.publish_topic, CompressedImage, queue_size= 1)
    seq_imgs = imgs_buffer(args.depth_camera_path, args.interval, (args.img_shape[0],args.img_shape[1] , args.observe_size))
    while not rospy.is_shutdown():
        if (seq_imgs.data.shape[-1]==args.observe_size):
            publisher.publish(seq_imgs.compress())
            seq_imgs.rate.sleep()

if __name__ =='__main__': 
    rospy.init_node('img_convert')
    main(parser())
