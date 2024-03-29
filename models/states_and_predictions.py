#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt, image as mpimg
import tensorflow as tf
import numpy as np
from model_train import load_data, generate_image
import os
from scipy.io import savemat, loadmat
import sys




def read_prediction():
    file_path = '{}/read me.txt'.format(model_name)
    with open(file_path, 'r') as f:
        for line in f:
            if 'Prediction gap' in line:
                inx = line.index('Prediction gap')+ len('Prediction gap')+1
                return int(line[inx])
    return -1


def add_border(img, border_size = 1, intense = 1):
    img_size = img.shape
    new_img = np.ones((img_size[0]+border_size*2,img_size[1]+border_size*2 ),dtype=float)*intense
    new_img[border_size:(border_size+img_size[0]),border_size:(border_size + img_size[1])] = img
    return new_img


if __name__ == '__main__':
    if len(sys.argv) > 2:
        observe_size = int(sys.argv[2])
    else:
        observe_size = 10

    model_name = sys.argv[1]
    img_seq = load_data(data_set_path = os.path.abspath(__file__ + '/../../') + "/data_set/test/")        
    generator = tf.keras.models.load_model('{}/generator'.format(model_name))
    input_size = generator.input.shape[3]
    GAP_PREDICT = read_prediction()
    assert GAP_PREDICT!=-1, "Prediction gap not found in 'read me' file."
    observe_size+= GAP_PREDICT
    start_inx = tf.random.uniform([1],0,img_seq.shape[2]-GAP_PREDICT-observe_size, dtype = tf.dtypes.int32).numpy()[0]

    x_real = add_border(img_seq[...,start_inx])
    img_shape = x_real.shape
    x_predict = np.ones(img_shape)
    x_predict_2 = np.ones(img_shape)
    gen_img = 0  # To avoid an error.
    
    for i in range(1,observe_size):
        x_real = np.concatenate((x_real, add_border(img_seq[...,start_inx+i])), axis = 1)
        x_predict = np.concatenate((x_predict, np.ones(img_shape)), axis = 1)
        x_predict_2 = np.concatenate((x_predict_2, np.ones(img_shape)), axis = 1)
        if i>=input_size+1 and GAP_PREDICT==0:
            rec_img = np.concatenate((img_seq[...,start_inx+i-input_size:start_inx+i-1],gen_img[...,tf.newaxis]), axis = 2)
            gen_img_2 = generator(rec_img[tf.newaxis,...], training = False)[0,...,0]
            x_predict_2[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img_2)
            gen_img = generator(img_seq[tf.newaxis,:,:,start_inx+i-input_size:start_inx+i], training = False)[0,...,0]
            x_predict[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img)
        elif i>=input_size + GAP_PREDICT:
            gen_img = generator(img_seq[tf.newaxis,:,:,start_inx+i-input_size-GAP_PREDICT:start_inx+i-GAP_PREDICT], training = False)[0,...,0]
            x_predict[:,img_shape[1]*i:(img_shape[1]*(i+1))] = add_border(gen_img)
    plt.figure()
    full_img = np.concatenate((x_real,x_predict,x_predict_2),axis = 0)
    if GAP_PREDICT!=0: full_img = full_img[:img_shape[0]*2,:]
    plt.imshow(full_img, cmap = 'gray',vmin= -1, vmax = 1)
    plt.axis('off')
    img_name = '{}/sample-{} model-{}.png'.format(model_name, start_inx+1, model_name)
    mpimg.imsave(img_name, full_img, cmap = 'gray')
    print("\nThe image '{}' was created.\n".format(img_name))
