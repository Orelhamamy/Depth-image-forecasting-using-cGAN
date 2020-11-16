#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt

def loaddata():
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    
    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',origin=(_URL), extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip),'facades/')
    return PATH

def load(img_file):
    image = tf.io.read_file(img_file)
    image = tf.image.decode_jpeg(image)
    
    width = tf.shape(image)[1]
    width = width // 2
    
    real_img = image[:,:width,:]
    input_img = image[:,width:,:]
    
    real_img = tf.cast(real_img, tf.float32)
    input_img = tf.cast(input_img, tf.float32)
    
    return input_img, real_img

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

PATH  = loaddata()
## ---------- Display image from data set ----------
input_img, real_img = load(PATH + 'train/100.jpg')
# plt.figure()
# plt.imshow(real/255.0)
# plt.figure()
# plt.imshow(in_img/255.0)

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_img = tf.stack([input_image, real_image], axis=0)
    stacked_img = tf.image.random_crop(stacked_img, size = [2 , IMG_HEIGHT, IMG_WIDTH, 3])
    return stacked_img[0], stacked_img[1]

def normalize(input_image, real_image):
    input_image = input_image/127.5 - 1
    real_image = real_image/127.5 - 1
    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    
    input_image, real_image = random_crop(input_image, real_image)
    
    if tf.random.uniform(())>0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    
    return input_image, real_image

## -------------Print 4 example----------------
# plt.figure(figsize = (6,6))
# for i in range(4):
#     inp_jit, re_jit = random_jitter(input_img, real_img)
#     plt.subplot(2,2, i+1)
#     plt.imshow(inp_jit/255.0)
#     plt.axis('off')
# plt.show()

def load_image_train(img_file):
    input_img, output_image = load(img_file)
    input_img, output_image = random_jitter(input_img, output_image)
    input_img, output_image = normalize(input_img, output_image)
    return input_img, output_image

def load_image_test(img_file):
    input_img, output_image = load(img_file)
    input_img, output_image = resize(input_img, output_image, IMG_HEIGHT, IMG_WIDTH)
    input_img, output_image = normalize(input_img, output_image)
    return input_img, output_image

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHENNELS = 3

def downsample(filters, size, apply_batchmorm = True):
    initializer = tf.random_normal_initializer(0.,0.02)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchmorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    
    return result    

def upsample(filters, size, apply_dropout = False):
    initializer = tf.random_normal_initializer(0.,0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides = 2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(.5))
        
    result.add(tf.keras.layers.ReLU())
    
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape =[IMG_HEIGHT, IMG_WIDTH , 3])
    
    down_stack = [
        downsample(64,4,apply_batchmorm=False), # (bs,128,128,64)
        downsample(128,4), # (bs,64,64,128)
        downsample(256,4), # (bs, 32,32,256)
        downsample(512,4), # (bs, 16,16,512)
        downsample(512,4), # (bs, 8,8,512)
        downsample(512,4), # (bs, 4,4,512)
        downsample(512,4), # (bs, 2,2,512)
        downsample(512,4)] # (bs, 1,1,512)
    
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2,2,512)
        upsample(512, 4, apply_dropout=True), # (bs, 4,4,512)
        upsample(512, 4, apply_dropout=True), # (bs, 8,8,512)
        upsample(512, 4), # (bs, 16,16,512)
        upsample(256,4), # (bs, 32,32,512)
        upsample(128,4), # (bs, 64,64,512)
        upsample(64,4)] # (bs, 128,128,512)
    
    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHENNELS, 4,strides=2,padding='same',
                                           activation ='tanh', 
                                           kernel_initializer= initializer,
                                           use_bias=False)
    
    x = inputs
    connections = []
    for down in down_stack:
        x = down (x)
        connections.append(x)
    connections = reversed(connections[:-1])
    
    for up, conc in zip(up_stack, connections):
        x = up (x)
        x = tf.keras.layers.Concatenate()([conc, x])
    x = last(x)
    
    return tf.keras.Model(inputs = inputs, outputs = x)

generetor = Generator()
tf.keras.utils.plot_model(generetor, show_shapes=True, dpi=64)
    
    

    



    