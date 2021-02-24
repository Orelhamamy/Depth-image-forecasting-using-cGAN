#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt, image as mpimg
import tensorflow as tf
import numpy as np
from model_train_base import load_data_with_future_y_train, generate_image
import os
from scipy.io import savemat, loadmat

model_name = 'cGAN_3pic_1y_train_2.4'

x_train, x_test, y_train, y_test = load_data_with_future_y_train(0.3)        
generator = tf.keras.models.load_model(model_name+'/generator_0')


def add_border(img , border_size = 1, intense = 1):
    img_size = img.shape
    new_img = np.ones((img_size[0]+border_size*2,img_size[1]+border_size*2 ),dtype=float)*intense
    new_img[border_size:(border_size+img_size[0]),border_size:(border_size + img_size[1])] = img
    return new_img

def get_x_y(image_number):
    if image_number in file_list_train:
        inx = file_list_train.index(image_number)
        return x_train[inx], y_train[inx]
    inx = file_list_test.index(image_number)
    return x_test[inx], y_test[inx]
        
        
file_lists = loadmat('{}/file_lists.mat'.format(model_name))
get_number = lambda x: [int(i) for i in x[:,1]]

file_list_test = get_number(file_lists['file_list_test'])
file_list_train = get_number(file_lists['file_list_train'])

start_inx = tf.random.uniform([1],0,len(x_train)+len(x_test)-10, dtype = tf.dtypes.int32).numpy()
input_gen, target = get_x_y(start_inx)
x_real = add_border(target[...,0])
x_predict = np.ones(x_real.shape)
x_predict_2 = np.ones(x_real.shape)
img_shape = x_real.shape
for i in range(1,10):
    input_gen, target = get_x_y(start_inx+i)
    x_real = np.concatenate((x_real, add_border(target[...,0])), axis = 1)
    x_predict = np.concatenate((x_predict, np.ones(img_shape)), axis = 1)
    x_predict_2 = np.concatenate((x_predict_2, np.ones(img_shape)), axis = 1)
    if i>=4:
        rec_img = np.concatenate((input_gen[...,1:],gen_img[...,tf.newaxis]), axis = 2)
        gen_img_2 = generator(rec_img[tf.newaxis,...], training = False)[0,...,0]
        x_predict_2[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img_2)
        gen_img = generator(input_gen[tf.newaxis,...], training = False)[0,...,0]
        x_predict[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img)
        
    elif i>=3:
        gen_img = generator(input_gen[tf.newaxis,...], training = False)[0,...,0]
        x_predict[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img)
plt.figure()
full_img = np.concatenate((x_real,x_predict,x_predict_2),axis = 0)
plt.imshow(full_img, cmap = 'gray',vmin= -1, vmax = 1)
plt.axis('off')
mpimg.imsave('{}/sample-{}.png'.format(model_name, start_inx), full_img, cmap = 'gray')
    

