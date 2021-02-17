#!/usr/bin/env python

from getkey import getkey, keys
import rospy
from geometry_msgs.msg import Twist
import time

rospy.init_node('teleop_by_arrows')
pub_vel_cmd = rospy.Publisher('cmd_vel', Twist, queue_size = 1)



def stop(pub, msg): # msg - geometry_msg - Twist
    msg.linear.x = 0 
    msg.angular.z = 0
    pub.publish(msg)
    
def move(linear, angular):
    global pub_vel_cmd
    msg = Twist()
    msg.linear.x = linear 
    msg.angular.z = angular 
    pub_vel_cmd.publish(msg)
    # time.sleep(.75)
    # stop(pub_vel_cmd, msg)
    
        
def listener():
    global senstive
    key = getkey()
    linear = 0; angular = 0
    while not rospy.is_shutdown():
        while not key:
            True
        if key==keys.UP:
            linear+=0.5
        elif key==keys.DOWN:
            linear-=0.5
        elif key==keys.RIGHT:
            angular-=.3
        elif key==keys.LEFT:
            angular+=.3
        elif key=='q':
            linear *=1.3; angular*=1.3
        elif key=='a':
            linear *=0.5; angular*=0.5
        elif key =='s':
            linear = 0; angular = 0
        move(linear, angular)
        key = getkey()
        
if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
    

    