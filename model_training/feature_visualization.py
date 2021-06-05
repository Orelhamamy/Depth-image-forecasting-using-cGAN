#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from model_train_base import load_data
from three_d_conv_model import Three_d_conv_model
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

def plot_2D(data_set_path, model_name):
    model_name = '--'
    output_layers = [1, 2, 3, 4]
    data = load_data(data_set_path)
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


def plot_3D_conv(test_set_path, model_name):
    model = Three_d_conv_model(model_name =model_name,data_set_path = test_set_path,
                               load_model = True)
    output_layers = [1, 2, 3, 4]
    inx = 169
    outputs = [model.generator.layers[i].output for i in output_layers]
    visual_gen = tf.keras.models.Model(inputs = model.generator.inputs, outputs = outputs)
    f_maps = visual_gen.predict(model.train_sequence[tf.newaxis, :,:,inx:inx+model.OBSERVE_SIZE, tf.newaxis])
    row = model.OBSERVE_SIZE
    for layer, feature_map in enumerate(f_maps):
        col = feature_map.shape[-1]
        ix = 1
        plt.figure()
        gs1 = gridspec.GridSpec(row, col)
        gs1.update(wspace = 0.01, hspace = 0.01)
        for j in range(col): # columns
            for i in range(row): # rows
                ax = plt.subplot2grid((row, col),(i,j))
                ax.axis('off')
                plt.imshow(feature_map[0,:,:,i,j], cmap= 'gray')
                ix += 1
        plt.show()
        plt.savefig('{}/feature-{}.png'.format(model_name, layer+1), bbox_inches='tight')
    plt.figure()
    model.generate_images(inx, model.generator, save = False)

    
if __name__ =='__main__':
    test_set_path = '/home/lab/orel_ws/project/data_set_armadillo/2/'
    model_name =  'ARM-3D_conv'
    plot_3D_conv(test_set_path, model_name)
    