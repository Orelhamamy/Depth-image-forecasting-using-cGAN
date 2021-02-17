#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model_train_base import load_data
#from scipy.io import savemat
#import datetime


def generate_image(imgs, input_size, save =False):
    if save: 
        path = os.path.split(save)[0]
        if not os.path.exists(path):
            os.makedirs(path)
    
    for i in range(input_size):
        axs = plt.subplot(1,input_size,i+1)
        axs.imshow(imgs[:,:,i], cmap = 'gray',  vmin = -1, vmax = 1)
        plt.title(i+1)
        plt.axis('off')
    if not save:
        plt.show()
    else:
        plt.savefig(save)

if __name__ =='__main__':
    
    model_name = 'cGAN_5pic_1y_train_1.0'
    output_layers = [1, 2, 3]
    data = load_data()
    inx = 10
    
    try: 
        visual_gen = tf.keras.models.load_model(model_name+'/generator')
    except OSError:
        exit
    outputs = [visual_gen.layers[i].output for i in output_layers]
    visual_gen = tf.keras.models.Model(inputs = visual_gen.inputs, outputs = outputs)
    input_size = visual_gen.inputs[0].shape[3]
    f_maps = visual_gen.predict(data[tf.newaxis, :,:,inx:inx+input_size])
    for feature_map in f_maps:
        dim = int(np.sqrt(feature_map.shape[3]))
        ix = 1
        plt.figure()
        for _ in range(dim): # columns
            for _ in range(dim): # rows
                ax = plt.subplot(dim, dim, ix)
                ax.axis('off')
                plt.imshow(feature_map[0,:,:,ix-1], cmap= 'gray')
                ix += 1
        plt.show()
    plt.figure()
    generate_image(data[:,:,inx:inx+input_size], input_size)
'''
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()
'''