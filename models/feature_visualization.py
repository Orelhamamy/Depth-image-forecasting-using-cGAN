#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from logging import error
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
sys.path.append('../')
from model_train import load_data
from three_d_conv_model import Three_d_conv_model
# from scipy.io import savemat
# import datetime


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

    output_layers = [1, 2]
    data = load_data(test_set_path)

    try:
        visual_gen = tf.keras.models.load_model(model_name+'/generator')
    except OSError:
        exit
    outputs = [visual_gen.layers[i].output for i in output_layers]
    visual_gen = tf.keras.models.Model(inputs = visual_gen.inputs, outputs = outputs)
    input_size = visual_gen.inputs[0].shape[3]
    f_maps = visual_gen.predict(data[tf.newaxis, :,:,inx:inx+input_size])
    dpi = 100
    fig_width = 8.3
    for layer, feature_map in enumerate(f_maps):
        dim = np.int(np.ceil(np.sqrt(feature_map.shape[3])))
        nrow = ncol = dim
        if layer!=0:
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
        #plt.show()
        plt.savefig('{}{}_feature-{}.png'.format(save_path, model_name, layer+1), dpi=dpi)
        plt.savefig('{}{}_feature-{}.eps'.format(save_path, model_name, layer+1), dpi=dpi, format='eps',pad_inches=0.0)
        plt.close()
    plt.figure()
    # generate_image(data[:,:,inx:inx+input_size], input_size)  # This will display the input.



def plot_3D_conv(test_set_path, model_name, inx, save_path = ''):
    if save_path=='': save_path = model_name+'/'
    model = Three_d_conv_model(model_name =model_name,data_set_path = test_set_path,
                                load_model = True)
    output_layers = [1, 2]
    outputs = [model.generator.layers[i].output for i in output_layers]
    visual_gen = tf.keras.models.Model(inputs = model.generator.inputs, outputs = outputs)
    f_maps = visual_gen.predict(model.train_sequence[tf.newaxis, :,:,inx:inx+model.OBSERVE_SIZE, tf.newaxis])
    row = model.OBSERVE_SIZE
    for layer, feature_map in enumerate(f_maps):
        col = feature_map.shape[-1]
        dpi = 100
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
        plt.close()
    plt.figure()
    #model.generate_images(inx, model.generator, save = False)

    
if __name__ =='__main__':
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        print("Error: Try again with entering the model name as arg.")
        exit()
    root_path = os.path.abspath(__file__ + "/../..")
    test_set_path = root_path + "/data_set/test/"
    model_path =  root_path + "/models/" + model_name
    if not os.path.exists(model_path):
        print("Error: The model {} don't exists.".format(model_name))
        exit()
    save_path = model_path + "/features/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    inx = 178 if "SM" in model_path else 195 # Choose the inx according to yours dataset. 
    if "3D" in model_path:
        plot_3D_conv(test_set_path, model_name, inx, save_path)
    else:
        if "Gap" in model_path: inx-=2
        plot_2D(test_set_path,model_name, inx, save_path)
    plt.close('all')
