#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-


import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16
import numpy as np
from datetime import datetime
import os


INTERVAL = 0.25
HEIGHT, WIDTH = 128, 128
DIFF = 1000 # Exam the diff between two sequential img 
depth_camera = '/depth_camera/depth/image_raw'
#data_set_path = '/home/lab/orel_ws/project/data_set/test'
data_set_path = '/home/lab/orel_ws/project/src/simulation_ws/data_set'
datetime = datetime.now()
date = datetime.now().strftime("%d-%m-%y,%H:%M")
MAX_LIN_VEL = 3 # maximum linear vel
MAX_ANG_VEL = 1.5 # 

class image_buffer():
    def __init__(self, img, bridge):
        img = cv2.resize(img,(WIDTH,HEIGHT))
        self.img = img *10
        self.bridge = bridge
        self.count = 1
        self.vel = Twist()
        try: 
            os.mkdir(data_set_path + '/vel_imgs')
        except OSError as e:
            pass
                  
        
    def callback(self, event):
        img_msg = rospy.wait_for_message(depth_camera,Image)
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            self.img = cv2.resize(img, (WIDTH, HEIGHT))*10
        except CvBridgeError:
            pass
        self.save_img(data_set_path, date)
    
    def save_img(self, path='', name='img'):
        print('Save img')
        cv2.imwrite('{}/{}--{}.jpg'.format(path, name, self.count), self.img)
        self.gen_vel_img(path, name)
        self.count+=1
    
    def gen_vel_img(self, path='', name = 'img'):
        path = path + '/vel_imgs'
        img = np.zeros((HEIGHT,WIDTH), dtype = 'int8')
        img[0:HEIGHT//2,:] = (self.vel.linear.x/MAX_LIN_VEL)*127.5
        img[:,0:WIDTH//2] += int((self.vel.angular.z/MAX_ANG_VEL)*127.5)
        img = img + abs(np.min(img))
        # print(img)
        # cv2.imshow('123', img)
        # cv2.waitKey(30)
        cv2.imwrite('{}/{}--{}.jpg'.format(path,name, self.count),img)
    
    def vel_callback(self, msg):
        self.vel = msg
        
        
def main():
    img_msg = rospy.wait_for_message(depth_camera,Image)
    bridge = CvBridge()
    if not os.path.exists(data_set_path):
        os.makedirs(data_set_path)
    img1 = image_buffer(bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough'), bridge)
    # img1.save_img(data_set_path, date)
    rospy.Subscriber('/cmd_vel', Twist,img1.vel_callback)
    rospy.Timer(rospy.Duration(INTERVAL), img1.callback)
    '''    
    while not rospy.is_shutdown():
        reff = img1.img
        diffrent = abs(np.sum(reff-img1.img))
        while(diffrent<DIFF):
            try:
                diffrent = abs(np.sum(reff - img1.img))
                if diffrent >10:
                    print(diffrent)
            except rospy.ROSInterruptException:
                    break
    '''
    rospy.spin()        
    


if __name__ =='__main__':
    try:
        rospy.init_node('data_set_creator')
        main()  
    except KeyboardInterrupt:
        pass