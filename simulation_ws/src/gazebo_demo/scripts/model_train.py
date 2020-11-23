#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

data_set_path = '/home/lab/orel_ws/project/simulation_ws/data_set'
train_dataset = tf.data.Dataset.list_files(data_set_path + '/*.jpg')

BUFFER_SIZE = tf.shape(os.listdir(data_set_path))-1
BATCH_SIZE = 1
HEIGHT, WIDTH = 128, 128
EPOCHS = 10
'''
@tf.autograph.experimental.do_not_convert
def load_image(img_file):
    type(img_file)
    name = str(img_file)
    image = tf.io.read_file(img_file)
    image = tf.image.decode_jpeg(image)
    # cv2.imshow('12', image)
    # num = int(name[name.index('--')+2:name.index('.jpg')])
    
    return image #, num 

train_dataset = train_dataset.map(load_image,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
'''
def read_img(name, start_img):
    imgs = []
    for img in range(10):
        x = tf.keras.preprocessing.image.load_img(data_set_path+'/'+name+str(start_img+img)+'.jpg')
        imgs.append(x)
    output = tf.keras.preprocessing.image.load_img(data_set_path+'/'+name+str(start_img+10)+'.jpg')
    return imgs, output
    
    
def load_data():
    name = os.listdir(data_set_path)[0]
    name = name[:name.index('--')+2]
    x_train =[]
    y_train =[]
    for i in range(BUFFER_SIZE.numpy()[0]-11):
        x, y = read_img(name,i+1)
        x_train.append(x)
        y_train.append(y)
    return x_train, y_train

x_train, y_train = load_data()        
    
    