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

def plot_2D(test_set_path, model_name, inx, save_path = ''):
    if save_path=='': save_path = model_name+'/'
    
    output_layers = [1, 2, 3, 4]
    data = load_data(test_set_path)
    
    try: 
        visual_gen = tf.keras.models.load_model(model_name+'/generator')
    except OSError:
        exit
    outputs = [visual_gen.layers[i].output for i in output_layers]
    visual_gen = tf.keras.models.Model(inputs = visual_gen.inputs, outputs = outputs)
    input_size = visual_gen.inputs[0].shape[3]
    f_maps = visual_gen.predict(data[tf.newaxis, :,:,inx:inx+input_size])
    dpi = 200
    fig_width = 8.3
    for layer, feature_map in enumerate(f_maps):
        dim = np.int(np.ceil(np.sqrt(feature_map.shape[3])))
        nrow = np.ceil(dim/2)
        ncol = dim*2
        ix = 1
        fig_height = fig_width*nrow/ncol
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        while(ix<=feature_map.shape[3]):
            ax = plt.subplot(nrow, ncol, ix)
            ax.axis('off')
            plt.imshow(feature_map[0,:,:,ix-1], cmap= 'gray')
            ix += 1
        plt.subplots_adjust(top = 1, bottom = 0, left = 0, right =1 ,wspace= .05, hspace = .05)
        plt.show()
        plt.savefig('{}{}_feature-{}.png'.format(save_path, model_name, layer+1), dpi=dpi)
        plt.savefig('{}{}_feature-{}.eps'.format(save_path, model_name, layer+1), dpi=dpi, format='eps',pad_inches=0.0)
    plt.figure()
    generate_image(data[:,:,inx:inx+input_size], input_size)


def plot_3D_conv(test_set_path, model_name, inx, save_path = ''):
    if save_path=='': save_path = model_name+'/'
    model = Three_d_conv_model(model_name =model_name,data_set_path = test_set_path,
                                load_model = True)
    output_layers = [1, 2, 3, 4]
    outputs = [model.generator.layers[i].output for i in output_layers]
    visual_gen = tf.keras.models.Model(inputs = model.generator.inputs, outputs = outputs)
    f_maps = visual_gen.predict(model.train_sequence[tf.newaxis, :,:,inx:inx+model.OBSERVE_SIZE, tf.newaxis])
    row = model.OBSERVE_SIZE
    for layer, feature_map in enumerate(f_maps):
        col = feature_map.shape[-1]
        dpi = 200
        fig_width = 8.3
        fig_height = fig_width*row/col
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        for j in range(col): # columns
            for i in range(row): # rows
                ax = plt.subplot2grid((row, col),(i,j))
                ax.axis('off')
                plt.imshow(feature_map[0,:,:,i,j], cmap= 'gray')
        plt.show()
        plt.subplots_adjust(top = 1, bottom = 0, left = 0, right =1 ,wspace= .05, hspace = .05)
        plt.savefig('{}{}_feature-{}.png'.format(save_path, model_name, layer+1), dpi=dpi)
        plt.savefig('{}{}_feature-{}.eps'.format(save_path, model_name, layer+1), dpi=dpi, format='eps')
    plt.figure()
    model.generate_images(inx, model.generator, save = False)

    
if __name__ =='__main__':
    test_set_path = '/home/lab/orel_ws/project/data_set_armadillo/2/'
    model_name =  'ARM-Recursive'
    save_path = '/home/lab/orel_ws/project/results/features/'
    if "3D" in model_name:
        plot_3D_conv(test_set_path, model_name, 196, save_path)
    else:
        plot_2D(test_set_path,model_name, 178, save_path)
    plt.close('all')
