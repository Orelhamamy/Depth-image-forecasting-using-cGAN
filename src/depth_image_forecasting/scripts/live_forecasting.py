#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import cv2
import numpy as np
import rospy
import copy
import os
import argparse
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont



vel_cmd = Twist()


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_topic",
        default="/depth_camera/depth/image_raw/Compressed",
        help="The listen topic by the model.",
    )
    parser.add_argument(
        "--model_name", default="SM-3D_conv", help="The name of the loaded model."
    )
    parser.add_argument(
        "--model_path",
        default=os.path.abspath(__file__ + "/../../../../") + "/models/",
        help="The model dirctory.",
    )
    return parser.parse_args()


def vel_callback(msg):
    global vel_cmd
    vel_cmd = msg


def adjust_img(img):
    # Adjust the image for generator model
    img = img / 127.5 - 1
    return img[tf.newaxis, ...]


def create_velocity_arrow(fig, canvas, ax):
    ax.clear()
    speed = np.linalg.norm(np.array([vel_cmd.angular.z, vel_cmd.linear.x]))
    arrow_width = speed*0.05 + .01
    ax.arrow(0.5, 0.25, -vel_cmd.angular.z, vel_cmd.linear.x,
             width=arrow_width, head_width=0.5*speed, head_length=0.5*speed,
             fc="k", ec="k")
    ax.axis("off")
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(-.25, 1.25)
    canvas.draw()
    img_shape = fig.canvas.get_width_height()[::-1] + (3,)
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
        img_shape)
    image_from_plot = cv2.resize(image_from_plot, (256, 256))
    grey_img = np.dot(image_from_plot[..., :3], [0.2989, 0.5870, 0.1140])
    return grey_img


def create_avg_text(comp_time, img_shape,  model_name):

    img = Image.new('L', (img_shape), color=(255))
    d = ImageDraw.Draw(img)

    d.text((10, 128), "Computation time:{:.3f}(sec)\nModel: {}".format(comp_time, model_name), fill=(0))

    return np.asarray(img)


def read_prediction(model_name):
    file_path = '{}/read me.txt'.format(model_name)
    with open(file_path, 'r') as f:
        for line in f:
            if 'Prediction gap' in line:
                inx = line.index('Prediction gap') + len('Prediction gap') + 1
                return int(line[inx])
    return -1


def main(args):
    model_path = args.model_path + args.model_name
    if not os.path.exists(model_path + "/generator"):
        print(("Couldn't find the trained model '{}', make sure "
               "you enter the correct model name as an argument "
               "and verified that is a trained model.").format(args.model_name))
        exit()
    elif "Gap" in args.model_name or read_prediction(model_path)!=0:
        print("This deployment can be executed only with the '3D' or 'Recursive' models...")
        exit()
    global vel_cmd
    fig = Figure(figsize=(512/100, 512/100))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    avg_calc = .0
    counter = 0
    ax.set_aspect("equal")
    generator = tf.keras.models.load_model("{}/generator".format(model_path))
    input_shape = generator.input.shape[1:4]
    input_imgs = rospy.wait_for_message(args.input_topic, CompressedImage)
    rospy.Subscriber("/cmd_vel", Twist, vel_callback)
    dividing_gap = np.ones((generator.input.shape[1], int(generator.input.shape[1] / 2)), dtype=np.float32)
    while not rospy.is_shutdown():
        start_time = rospy.Time.now()
        input_imgs = np.fromstring(input_imgs.data, np.float32).reshape(input_shape)
        input_imgs = adjust_img(input_imgs)
        current_frame = copy.copy(input_imgs[0, :, :, -1])
        prediction = generator(input_imgs, training=False)[0, ...]
        if "3D" in args.model_name:
            prediction = prediction[..., 0]
            rec_prediction = prediction[..., 1]
        else:
            input_imgs = np.concatenate((input_imgs[0, ..., 1:], prediction), axis=-1)
            rec_prediction = generator(input_imgs[tf.newaxis, ...], training=False)[0, ..., 0]

        comp_calc = (rospy.Time.now() - start_time).to_sec()
        current_frame = ((current_frame + 1) * 127.5) / 100
        current_frame[current_frame > 1] = 1
        display_img = np.concatenate((current_frame, dividing_gap,
                                     ((prediction[..., 0] + 1)),
                                     dividing_gap, (rec_prediction + 1),
                                      ), axis=1)
        arrow = create_velocity_arrow(fig, canvas, ax)/255
        compu_calc_image = create_avg_text(comp_calc, (256, 256), args.model_name)/255
        arrow = np.concatenate((compu_calc_image, arrow), axis=1)
        display_img = np.concatenate((display_img, arrow), axis=0)
        display_img = cv2.copyMakeBorder(display_img, top=50, bottom=0, left=20, right=20, borderType=cv2.BORDER_CONSTANT, value=[1, 1])
        cv2.imshow("prediction", display_img)
        cv2.waitKey(1)
        if (comp_calc < 0.1):
            avg_calc = (avg_calc*counter + comp_calc) / (counter+1)
            print("Average computation time for {}: {:.4f} (sec)".format(args.model_name, avg_calc))
            counter += 1

        input_imgs = rospy.wait_for_message(args.input_topic, CompressedImage)


if __name__ == "__main__":
    rospy.init_node("obstacle_avoidance")
    main(parser())
