#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import copy
from scipy.io import savemat, loadmat
'''
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
OBSERVE_SIZE = 5 # How many img to observe
Y_TRAIN_SIZE = 5 # How many img to learn recursive 1= without recursion
OUTPUT_SIZE = 1
LAMBDA = 100 # determine the weight of l1 in Loss function (generator) LAMBDA = 0 -> cancel l1_loss
loss_object  = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LR_GEN =2e-4; BETA_1_GEN =0.5; BETA_2_GEN =.999
LR_DISC =2e-4; BETA_1_DISC =0.5; BETA_2_DISC =.999
ITERATION_GEN = 1 ; ITERATION_DISC = 1
log_dir = 'logs/'
checkpoint_dir = './traning_checkpoints'
'''


class Three_d_conv_model():
    def __init__(self, data_set_path, model_name,load_model = False, OBSERVE_SIZE = 10,
                 Y_TRAIN_SIZE = 1, HEIGHT = 128, WIDTH = 128, ALPHA = 0.2, kernel_size = 3,
                 LAMBDA = 100, LR_GEN = 2e-4, BETA_1_GEN =0.5, BETA_2_GEN =.999, LR_DISC= 2e-4, BETA_1_DISC =0.5, 
                 BETA_2_DISC =.999, prediction_gap = 0, concate = True):
        
        file_num = lambda x: int(x[x.index('--')+2:x.index('.jpg')])
        self.initializer = tf.random_normal_initializer(0.,0.02) # Var= 0.02
        self.model_name = model_name
        if not load_model:
            self.OBSERVE_SIZE = OBSERVE_SIZE ; self.Y_TRAIN_SIZE = Y_TRAIN_SIZE
            self.HEIGHT = HEIGHT; self.WIDTH = WIDTH 
            self.kernel_size = kernel_size; 
            self.ALPHA = ALPHA # Alpha for leakyReLU
            self.LAMBDA = LAMBDA
            self.concate = concate
            self.data_set_path = data_set_path
            self.beta= [BETA_1_GEN, BETA_2_GEN, BETA_1_DISC, BETA_2_DISC]
            self.learning_rates = np.array([[LR_GEN],[LR_DISC]])
            self.prediction_gap = prediction_gap
            self.save_pram(model_name)

        else:
            self.generator = tf.keras.models.load_model(model_name + '/generator')
            self.discriminator = tf.keras.models.load_model(model_name + '/discriminator')
            self.load_parm(model_name)

        self.file_list = [[file, file_num(file)] for file in os.listdir(self.data_set_path) 
                  if str(file).endswith('.jpg')]
        self.file_list.sort(key = lambda x:x[1])
        try:
            self.discriminator_reff = tf.keras.models.load_model(model_name+'/discriminator_reff')
        except OSError:
            self.discriminator_reff = False
        
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.initializer = tf.random_normal_initializer(0.,0.02) # Var =0.02
        
        self.load_data()
        self.gen_optimizer = tf.keras.optimizers.Adam(self.learning_rates[0,0], beta_1 =self.beta[0], beta_2 = self.beta[1])
        self.disc_optimizer = tf.keras.optimizers.Adam(self.learning_rates[1,0], beta_1= self.beta[2], beta_2 = self.beta[3])
        
    def save_pram(self,model_name):
        dic = {'OBSERVE_SIZE': self.OBSERVE_SIZE,'Y_train_size':self.Y_TRAIN_SIZE, 'Height':self.HEIGHT,'Width':self.WIDTH, 'Alpha':self.ALPHA,
               'Kernel_size': self.kernel_size, 'Lambda':self.LAMBDA, 'Concate': self.concate,
               'Data_set_path': self.data_set_path, 'Learning_rates': self.learning_rates, 'Beta': self.beta,
               'Prediction_gap':self.prediction_gap}
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        savemat('{}/parameters.mat'.format(model_name),dic)
    
    def load_parm(self,model_name):
        mat_fname = '{}/parameters.mat'.format(model_name)
        param_dic = loadmat(mat_fname)
        self.OBSERVE_SIZE = param_dic['OBSERVE_SIZE'][0][0]; self.Y_TRAIN_SIZE = param_dic['Y_train_size'][0][0]
        self.HEIGHT = param_dic['Height'][0][0]; self.WIDTH = param_dic['Width'][0][0]
        self.ALPHA = param_dic['Alpha'][0][0] # Alpha for leakyReLU
        self.kernel_size = param_dic['Kernel_size'][0][0];
        self.LAMBDA = param_dic['Lambda'][0][0]
        self.concate = param_dic['Concate'][0][0]
        self.data_set_path = param_dic['Data_set_path'][0]
        self.learning_rates = param_dic['Learning_rates'][:,0]; self.beta=param_dic['Beta'][0]
        self.prediction_gap = param_dic['Prediction_gap'][0][0]
        
    def read_img(self, img_name):
        x = tf.keras.preprocessing.image.load_img(self.data_set_path + img_name, 
                                                  color_mode='grayscale')
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = x/127.5-1 
        return x
    
    def load_data(self):
        self.train_sequence = self.read_img(self.file_list[0][0])
        for img_name in self.file_list[1:]:
            self.train_sequence = np.concatenate((self.train_sequence , self.read_img(img_name[0])), axis=2)
        self.data_set_size = self.train_sequence.shape[2]

    
    def generate_images(self, inx, model = False, training = False, columns = 5, save = False):
        columns = self.OBSERVE_SIZE
        rows = 3 # input, target, prediction
        print(rows)
        prediction = self.generator(self.train_sequence[tf.newaxis,:,:,inx:inx + self.OBSERVE_SIZE, tf.newaxis], training= training)[0,...,0]
        for i in range(self.OBSERVE_SIZE):
            axs = plt.subplot(rows, columns, i+1)
            axs.imshow(self.train_sequence[...,inx+i], cmap = 'gray')
            plt.title(i+1)
            plt.axis('off')
            plt.subplot(rows, columns, columns + i+1)
            plt.imshow(self.train_sequence[...,inx+self.OBSERVE_SIZE+i+1], cmap = 'gray')
            plt.title('Output {}'.format(str(i)))
            plt.axis('off')
            if model:
                plt.subplot(rows, columns, 2*columns + i+1)
                plt.imshow(prediction[:,:,i], cmap = 'gray')
                plt.axis('off')
                plt.title('Predict')
        plt.subplot(rows, columns, 1)
        plt.xlabel('Input')
        plt.subplot(rows, columns, columns+ 1)
        plt.xlabel('Target')
        plt.subplot(rows, columns,2*columns + 1)
        plt.xlabel('Prediction')
        if not save:
            plt.show()
        else:
            plt.savefig(save)
        
    def downsample(self, filters, size, apply_batchnorm = False):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv3D(filters, size, strides=(2,2,1), padding='same',
                                          kernel_initializer=self.initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU(self.ALPHA))
        return result
    
    def upsample(self, filters, size, apply_batchnorm = False, apply_dropout = False):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv3DTranspose(filters, size, strides = (2, 2, 1),
                                                   padding = 'same', kernel_initializer = self.initializer,
                                                   use_bias = False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(apply_dropout))
        
        result.add(tf.keras.layers.ReLU())
        return result
    
    
    def create_generator(self):
        inputs = tf.keras.layers.Input(shape = (self.HEIGHT, self.WIDTH, self.OBSERVE_SIZE, 1))
        downing = [
            self.downsample(10, self.kernel_size),
            self.downsample(28, self.kernel_size),
            self.downsample(36, self.kernel_size),
            self.downsample(52, self.kernel_size),
            self.downsample(128, self.kernel_size),
            self.downsample(256, self.kernel_size),
            self.downsample(512, self.kernel_size)
            ]
        upping = [
            self.upsample(256, self.kernel_size, apply_dropout = 0.5),
            self.upsample(128, self.kernel_size, apply_dropout = 0.5),
            self.upsample(52, self.kernel_size, apply_dropout = 0.5),
            self.upsample(36, self.kernel_size),
            self.upsample(28, self.kernel_size),
            self.upsample(10, self.kernel_size)
            ]
        last = tf.keras.layers.Conv3DTranspose(1, self.kernel_size, strides = (2,2,1), 
                                               padding= 'same', activation = 'tanh', 
                                               kernel_initializer = self.initializer)
        x = inputs
        connections = []
        for down in downing:
            x = down (x)
            connections.append(x)
        connections = reversed(connections[:-1])
        
        for up, conc in zip(upping, connections):
            x = up (x)
            if self.concate:
                x = tf.keras.layers.Concatenate(axis = -1)([conc, x])
        x = last(x)
        
        self.generator = tf.keras.Model(inputs = inputs, outputs = x)
    
    def generator_loss(self, disc_gen_output, gen_output, target):
        
        gen_loss = self.loss_object(tf.ones_like(disc_gen_output),disc_gen_output) 
        l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
        tot_loss = gen_loss +self.LAMBDA * l1_loss
        return tot_loss, gen_loss, l1_loss
        
    def create_discriminator(self):
        
        in_imgs = tf.keras.layers.Input(shape = [self.HEIGHT, self.WIDTH, self.OBSERVE_SIZE, 1], name = 'input_imgs')
        tar_img = tf.keras.layers.Input(shape = [self.HEIGHT, self.WIDTH, self.OBSERVE_SIZE, 1], name = 'target_imgs')
        
        conc = tf.keras.layers.Concatenate(axis = 3)([in_imgs, tar_img])
        
        down1 = tf.keras.layers.Conv3D(32, self.kernel_size, strides = (2,2,2), kernel_initializer = self.initializer,
                                       padding = 'same', use_bias=False)(conc)
        down2 = tf.keras.layers.Conv3D(128, self.kernel_size, strides = (2,2,2), kernel_initializer = self.initializer, 
                                       padding='same', use_bias = False)(down1)
        conv1 = tf.keras.layers.Conv3D(256, self.kernel_size, strides = (1,1,2), kernel_initializer = self.initializer,
                                       padding = 'same', use_bias = False)(down2)
        conv2 = tf.keras.layers.Conv3D(512, self.kernel_size, strides = (1,1,2), kernel_initializer = self.initializer,
                                       padding = 'same', use_bias = False)(conv1)
        batch_norm = tf.keras.layers.BatchNormalization()(conv2)
        active_relu = tf.keras.layers.LeakyReLU(self.ALPHA)(batch_norm)
        last = tf.keras.layers.Conv3D(1, self.kernel_size, strides = (1,1,2), kernel_initializer = self.initializer,
                                      padding = 'same', use_bias = False)(active_relu)
        self.discriminator = tf.keras.Model(inputs = [in_imgs, tar_img], outputs = last)
    
    def discriminator_loss(self, disc_gen_output, disc_real_output):
        
        gen_loss = self.loss_object(tf.zeros_like(disc_gen_output),disc_gen_output)
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        
        return gen_loss + real_loss
   
    def train_step(self, input_imgs, target, ):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            gen_output  = self.generator(input_imgs, training = True)
            disc_gen_output = self.discriminator([input_imgs, gen_output[...,0]], training = True)
            gen_tot_loss, gen_loss, gen_l1_loss = self.generator_loss(disc_gen_output, gen_output[...,0], target)
            
            gradient_gen = gen_tape.gradient(gen_tot_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
       
            disc_real_output = self.discriminator([input_imgs, target], training = True)
            disc_loss = self.discriminator_loss(disc_gen_output, disc_real_output)
                
            gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))
            
            self.losses_val = np.append(self.losses_val,[[gen_tot_loss.numpy()],[gen_loss.numpy()],
                                                         [gen_l1_loss.numpy()],[disc_loss.numpy()]], axis = 1)
            
        
    def sample_imgs(self, epoch, model_name):
        self.generate_images(30, model = self.generator, save = '{}/figs/epoch--{}.png'.format(model_name,epoch+1))
        inx = tf.random.uniform([1],0,self.data_set_size - self.OBSERVE_SIZE-self.prediction_gap -1, dtype = tf.dtypes.int32).numpy()[0]
        self.generate_images(inx, model = self.generator, save = '{}/figs/random/epoch--{}_inx.png'.format(model_name,epoch+1,inx))
    
    
    def fit(self, epochs, model_name):
        self.losses = np.zeros((5,0))
        self.losses_val = np.zeros((4,0))
        if not os.path.exists(model_name+'/figs'):
            os.makedirs(model_name+'/figs')
        with open(model_name +'/read me.txt','a') as f:
               f.write('\n {} {}, Epochs:{}, Y_train (recursive):{}, Prediction gap:{}.'
                       .format(model_name, datetime.datetime.now().strftime('%Y.%m.%d--%H:%M:%S'), epochs,
                               self.Y_TRAIN_SIZE, self.prediction_gap))
               if not self.discriminator_reff:
                   f.write(', Reff.')
               f.close()
        self.sample_imgs(-1, model_name)
        for epoch in range(epochs):
            start = time.time()
            reff_loss = 0
            print('Epoch:', epoch+1)
            
            for img_inx in range(self.data_set_size-2*self.OBSERVE_SIZE-self.prediction_gap):
                target_inx = img_inx+self.OBSERVE_SIZE+self.prediction_gap+1
                input_seq = copy.copy(self.train_sequence[:,:,img_inx:img_inx+self.OBSERVE_SIZE])
                
                for rec in range(self.Y_TRAIN_SIZE):
                    target = self.train_sequence[:,:,target_inx+rec+self.prediction_gap:target_inx+rec+self.OBSERVE_SIZE + self.prediction_gap]
                    self.train_step(input_seq[tf.newaxis,...,tf.newaxis],
                                    target[tf.newaxis,...,tf.newaxis],
                                    epoch)
                    gen_img = self.generator(input_seq[tf.newaxis,...,tf.newaxis])
                    
                    if rec==0 and self.discriminator_reff:
                        disc_gen = self.discriminator_reff([input_seq[tf.newaxis,...,tf.newaxis], gen_img])
                        disc_real = self.discriminator_reff([input_seq[tf.newaxis,...,tf.newaxis],
                                                             target[tf.newaxis,...,tf.newaxis]])
                        reff_real_loss, reff_gen_loss = self.discriminator_loss(disc_gen, disc_real)
                        reff_loss += reff_gen_loss/(self.data_set_size-2*self.OBSERVE_SIZE-self.prediction_gap)
                    input_seq = tf.concat([input_seq, gen_img[0,...,0]], axis = 3)
                    input_seq = input_seq[:,:,1:]
                print('.')
                if (epoch+1)%100==0:
                    print('\n')                    
            self.losses = np.append(self.losses, np.append(self.losses_val.mean(axis=1),reff_loss, axis=0), axis = 1)
            if (epoch+1)%200==0:
                if not self.discriminator_reff:
                    self.generator.save(model_name+'/generator_0')
                    self.discriminator.save(model_name + '/discriminator_reff')
                else:
                    self.generator.save(model_name+'/generator')
                    self.discriminator.save(model_name + '/discriminator')
                    
            self.sample_imgs(epoch, model_name)
            print('\nTime taken to epoch: {} in sec {}\n'.format(epoch, time.time()-start))
        if self.discriminator_reff:
            self.generator.save(model_name+'/generator')
            self.discriminator.save(model_name + '/discriminator')
        else: 
            self.generator.save(model_name+'/generator_0')
            self.discriminator.save(model_name + '/discriminator_reff')
          

    def print_model(self):
        tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi = 96, to_file='{}/Generator.png'.format(self.model_name))
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi = 96, to_file='{}/Discriminator.png'.format(self.model_name))

# model = Three_d_conv_model('/home/lab/orel_ws/project/simulation_ws/data_set/','3D_conv_try',OBSERVE_SIZE=3,
#                             Y_TRAIN_SIZE=2,LR_GEN=2e-5,concate=False)
model_name = '3D_conv_try'
model = Three_d_conv_model('/home/lab/orel_ws/project/src/simulation_ws/data_set/', model_name)
model.create_generator()
model.create_discriminator()
model.print_model()
model.fit(5, model_name)

# model = Three_d_conv_model(data_set_path,Y_TRAIN_SIZE=1)
# model.create_generator()
# model.create_discriminator()
# model.generate_images(1,model = True)




