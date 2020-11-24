#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data_set_path = '/home/lab/orel_ws/project/simulation_ws/data_set/'
# train_dataset = tf.data.Dataset.list_files(data_set_path + '/*.jpg')
file_num = lambda x: int(x[x.index('--')+2:x.index('.jpg')])

file_list = [[file, file_num(file)] for file in os.listdir(data_set_path)
             if file.endswith('.jpg')]
file_list.sort(key = lambda x:x[1])
BUFFER_SIZE = len(file_list)
BATCH_SIZE = 1
HEIGHT, WIDTH = 128, 128
EPOCHS = 10
VAR = 0.02 # Variance of initialize kernels.
ALPHA = 0.2 # Alpha for leakyReLU
DROP_RATE = 0.5 # Dropout rate for upsample.
OBSERVE_SIZE = 10
OUTPUT_SIZE = 1

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
def read_img(files):
    imgs = []
    for img in range(OBSERVE_SIZE):
        x = tf.keras.preprocessing.image.load_img(data_set_path + files[img][0])
        imgs.append(x)
    output = tf.keras.preprocessing.image.load_img(data_set_path + files[-1][0])
    return imgs, output
    
    
def load_data():
    x_train =[]
    y_train =[]
    for i in range(BUFFER_SIZE-OBSERVE_SIZE-1):
        if file_list[i][1]+OBSERVE_SIZE==file_list[i+OBSERVE_SIZE][1]:
            x, y = read_img(file_list[i:i+OBSERVE_SIZE+1])
            x_train.append(x)
            y_train.append(y)
    return x_train, y_train

x_train, y_train = load_data()        
def generate_image(input_imgs, output_img, model = False, training = False):
    if model:
        prediction = model(input_imgs, training= training)
        plt.subplot(3,5,14)
        plt.imshow(prediction)
        plt.axis('off')
        plt.title('Predict')
    for i in range(OBSERVE_SIZE):
        axs = plt.subplot(3,5,i+1)
        axs.imshow(input_imgs[i])
        plt.title(i+1)
        plt.axis('off')
    plt.subplot(3,5,13)
    plt.imshow(output_img)
    plt.title('Output')
    plt.axis('off')
    plt.show()
    
# ----------------- Random generate imgs -------------------------    
inx = tf.random.uniform([1],0,len(x_train),dtype = tf.dtypes.int32).numpy()[0]
generate_image(x_train[inx],y_train[inx])

# ----------------------------------------------------------------

def downsample(filters, size, apply_batchnorm = True):
    initializer = tf.random_normal_initializer(0.,VAR)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,size,
                                      strides=2, padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU(ALPHA))
    
    return result

def upsample(filters, size, apply_batchnorm = False, apply_dropout = False):
    initializer = tf.random_normal_initializer(0.,VAR)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters,size,
                                      strides=2, padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(apply_dropout))
    
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[HEIGHT, WIDTH,OBSERVE_SIZE])
    
    downing = [ 
        downsample(32, 4, apply_batchnorm=False), # (bs, 64,64,32)
        downsample(64, 4), # (bs, 32,32,64)
        downsample(128, 4), # (bs, 16,16,128)
        downsample(256, 4), # (bs, 8,8,256)
        downsample(512, 4), # (bs, 4,4,512)
        downsample(512, 4), # (bs, 2,2,512)
        downsample(512, 4), # (bs, 1,1,512)
        ]
    
    upping = [
        upsample(512, 4, apply_dropout = DROP_RATE), # (bs, 2,2,512)
        upsample(512, 4, apply_dropout = DROP_RATE), # (bs, 4,4,512)
        upsample(256, 4, apply_dropout = DROP_RATE), # (bs, 8,8,256)
        upsample(128, 4), # (bs, 16,16,128)
        upsample(64, 4), # (bs, 32,32,64)
        upsample(32,4), # (bs, 64,64,32)
        ]
    initializer = tf.random_normal_initializer(0.,VAR)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_SIZE, 4, strides = 2,
                                           padding= 'same', activation = 'tanh',
                                           kernel_initializer = initializer)
    
    x = inputs
    connections = []
    for down in downing:
        x = down(x)
        connections.append(x)
    connections = reversed(connections[:-1])
    
    for up, conc in zip(upping, connections):
        x = up (x)
        x = tf.keras.layers.Concatenate()([conc , x])
    x = last(x)
    
    return tf.keras.Model(inputs = inputs, outputs = x)
generator = Generator()

tf.keras.utils.plot_model(generator, show_shapes=True, 
                          dpi = 96, to_file='Generator.png')