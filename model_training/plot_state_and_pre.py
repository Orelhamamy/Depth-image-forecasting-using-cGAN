#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt, image as mpimg
import tensorflow as tf
import numpy as np
from model_train_base import load_data, generate_image
import os
from scipy.io import savemat, loadmat

model_name = 'cGAN_5pic_1y_train_1.5'
observe_size = 10
img_seq = load_data()        
generator = tf.keras.models.load_model('{}/generator'.format(model_name))
input_size = generator.input.shape[3]

def add_border(img , border_size = 1, intense = 1):
    img_size = img.shape
    new_img = np.ones((img_size[0]+border_size*2,img_size[1]+border_size*2 ),dtype=float)*intense
    new_img[border_size:(border_size+img_size[0]),border_size:(border_size + img_size[1])] = img
    return new_img


start_inx = tf.random.uniform([1],0,img_seq.shape[2]-observe_size, dtype = tf.dtypes.int32).numpy()[0]
# start_inx = 5

x_real = add_border(img_seq[...,start_inx])
img_shape = x_real.shape
x_predict = np.ones(img_shape)
x_predict_2 = np.ones(img_shape)

for i in range(1,observe_size):
    x_real = np.concatenate((x_real, add_border(img_seq[...,start_inx+i])), axis = 1)
    x_predict = np.concatenate((x_predict, np.ones(img_shape)), axis = 1)
    x_predict_2 = np.concatenate((x_predict_2, np.ones(img_shape)), axis = 1)
    if i>=input_size+1:
        rec_img = np.concatenate((img_seq[...,start_inx+i+1-input_size:start_inx+i],gen_img[...,tf.newaxis]), axis = 2)
        # plt.figure() 
        # plt.imshow(rec_img.reshape((128,-1,1),order = 'F'),cmap='gray', vmin=-1, vmax=1)
        gen_img_2 = generator(rec_img[tf.newaxis,...], training = False)[0,...,0]
        x_predict_2[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img_2)
        gen_img = generator(img_seq[tf.newaxis,:,:,start_inx+i-input_size:start_inx+i], training = False)[0,...,0]
        x_predict[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img)
        
    elif i>=input_size:
        gen_img = generator(img_seq[tf.newaxis,:,:,start_inx+i-input_size:start_inx+i], training = False)[0,...,0]
        x_predict[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img)
plt.figure()
full_img = np.concatenate((x_real,x_predict,x_predict_2),axis = 0)
plt.imshow(full_img, cmap = 'gray',vmin= -1, vmax = 1)
plt.axis('off')

mpimg.imsave('{}/sample-{}.png'.format(model_name, start_inx+1), full_img, cmap = 'gray')
