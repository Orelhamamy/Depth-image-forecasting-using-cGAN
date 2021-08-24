#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-


import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
from datetime import datetime
import os
import sys
import shutil

INTERVAL = 0.25
HEIGHT, WIDTH = 128, 128
depth_camera = '/depth_camera/depth/image_raw'
datetime = datetime.now()
date = datetime.now().strftime("%d-%m-%y,%H:%M")
data_set_path = os.path.abspath(__file__ + '/../../../../') + "/data_set"
velocity_generate = False
MAX_LIN_VEL = 3  # maximum linear vel, used for normalization of the velocity img
MAX_ANG_VEL = 1.5


class image_buffer():
    def __init__(self, img, bridge):
        img = cv2.resize(img, (WIDTH, HEIGHT))
        self.img = img * 10
        self.bridge = bridge
        self.count = 1
        self.vel = Twist()
        if velocity_generate:
            try:
                os.mkdir(data_set_path + '/vel_imgs')
            except OSError:
                pass

    def callback(self, event):
        img_msg = rospy.wait_for_message(depth_camera, Image)
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            self.img = cv2.resize(img, (WIDTH, HEIGHT))*10
        except CvBridgeError:
            pass
        self.save_img(data_set_path, date)

    def save_img(self, path='', name='img'):
        print('Save img')
        cv2.imwrite('{}/{}--{}.jpg'.format(path, name, self.count), self.img)
        if velocity_generate:
            self.gen_vel_img(path, name)
        self.count += 1

    def gen_vel_img(self, path='', name='img'):
        path = path + '/vel_imgs'
        img = np.zeros((HEIGHT, WIDTH), dtype='int8')
        img[0:HEIGHT // 2, :] = (self.vel.linear.x / MAX_LIN_VEL) * 127.5
        img[:, 0:WIDTH // 2] += int((self.vel.angular.z / MAX_ANG_VEL) * 127.5)
        img = img + abs(np.min(img))
        cv2.imwrite('{}/{}--{}.jpg'.format(path, name, self.count), img)

    def vel_callback(self, msg):
        self.vel = msg


def main():
    img_msg = rospy.wait_for_message(depth_camera, Image)
    bridge = CvBridge()
    if os.path.exists(data_set_path):
        ans = raw_input("The creation of new data set will destroy the old, are you sure [y/n]: ")
        while(ans not in ['n', 'y']):
            ans = raw_input("Bad input [y/n]: ")
        if ans == 'n':
            print('Exit')
            return
        shutil.rmtree(data_set_path)
    os.makedirs(data_set_path)
    img1 = image_buffer(bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough'), bridge)
    rospy.Subscriber('/cmd_vel', Twist, img1.vel_callback)
    rospy.Timer(rospy.Duration(INTERVAL), img1.callback)
    rospy.spin()


if __name__ == '__main__':
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) < 2 or all(myargv[1] != i for i in ["test", "train"]):
        print("Missed usage arg\ndata_set_creator usage \nusage can be: train or test")
    else:
        data_set_path = data_set_path + "/{}".format(myargv[1])
        rospy.init_node('data_set_creator')
        main()
