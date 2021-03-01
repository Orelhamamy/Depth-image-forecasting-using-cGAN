#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
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
Y_TRAIN_SIZE = 5 # How many img to learn recursive
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
    def __init__(self, data_set_path, model_name,load_model = False, OBSERVE_SIZE = 5,
                 Y_TRAIN_SIZE = 1, HEIGHT = 128, WIDTH = 128, ALPHA = 0.2, kernel_size = 3, OUTPUT_SIZE = 1,
                 LAMBDA = 100, LR_GEN = 2e-4, BETA_1_GEN =0.5, BETA_2_GEN =.999, LR_DISC= 2e-4, BETA_1_DISC =0.5, BETA_2_DISC =.999,
                 ITERATION_GEN = 1 ,ITERATION_DISC = 1, log_dir = 'logs/' , concate = True):
        self.x_train =[]; self.y_train =[]
        file_num = lambda x: int(x[x.index('--')+2:x.index('.jpg')])
        self.initializer = tf.random_normal_initializer(0.,0.02) # Var= 0.02
        if not load_model:
            self.OBSERVE_SIZE = OBSERVE_SIZE ; self.Y_TRAIN_SIZE =Y_TRAIN_SIZE
            self.HEIGHT = HEIGHT; self.WIDTH = WIDTH 
            self.ALPHA = ALPHA # Alpha for leakyReLU
            self.kernel_size = kernel_size; self.output_size = OUTPUT_SIZE
            self.LAMBDA = LAMBDA
            self.concate = concate
            self.data_set_path = data_set_path
            self.log_dir = log_dir
            self.iteration_gen = ITERATION_GEN ; self.iteration_disc = ITERATION_DISC       
            self.create_generator()
            self.create_discriminator()
            self.lr_gen = LR_GEN; self.lr_disc = LR_DISC; self.beta= [BETA_1_GEN, BETA_2_GEN, BETA_1_DISC, BETA_1_GEN]
            self.save_pram(model_name)
            self.file_list = [[str(file), file_num(file)] for file in os.listdir(self.data_set_path)
                  if file.endswith('.jpg')]
            
        else:
            # self.generator = tf.keras.models.load_model(model_name + '/generator')
            # self.discriminator = tf.keras.models.load_model(model_name + '/discriminator')
            self.load_parm(model_name)

        try:
            files = [ f.decode('utf-8') for f in os.listdir(self.data_set_path)]
        except AttributeError:
            files = os.listdir(self.data_set_path)
        self.file_list = [[file, file_num(file)] for file in files
                  if file.endswith('.jpg')]
        self.file_list.sort(key = lambda x:x[1])
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.initializer = tf.random_normal_initializer(0.,0.02) # Var =0.02

        self.load_data_with_future_y_train(Y_TRAIN_SIZE)
        self.gen_optimizer = tf.keras.optimizers.Adam(LR_GEN, beta_1 =self.beta[0], beta_2 = self.beta[1])
        self.disc_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1= self.beta[2], beta_2 = self.beta[3])
        
    def save_pram(self,model_name):
        dic = {'OBSERVE_SIZE': self.OBSERVE_SIZE,'Y_train_size':self.Y_TRAIN_SIZE, 'Height':self.HEIGHT,'Width':self.WIDTH, 'Alpha':self.ALPHA,
               'Kernel_size': self.kernel_size,'Output_size':self.output_size, 'Lambda':self.LAMBDA, 'Concate': self.concate,
               'Data_set_path': self.data_set_path, 'Log_dir':self.log_dir, 'Iteration_gen':self.iteration_gen,
               'Iteration_disc': self.iteration_disc, 'Learning_rate': [self.lr_gen, self.lr_disc],
               'Beta': self.beta}
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        savemat('{}/parameters.mat'.format(model_name),dic)
    
    def load_parm(self,model_name):
        mat_fname = '{}/parameters.mat'.format(model_name)
        param_dic = loadmat(mat_fname)
        self.OBSERVE_SIZE = param_dic['OBSERVE_SIZE'][0][0]; self.Y_TRAIN_SIZE = param_dic['Y_train_size'][0][0]
        self.HEIGHT = param_dic['Height'][0][0]; self.WIDTH = param_dic['Width'][0][0]
        self.ALPHA = param_dic['Alpha'][0][0] # Alpha for leakyReLU
        self.kernel_size = param_dic['Kernel_size'][0][0]; self.output_size = param_dic['Output_size'][0][0]
        self.LAMBDA = param_dic['Lambda'][0][0]
        self.concate = param_dic['Concate'][0][0]
        self.data_set_path = param_dic['Data_set_path'][0]
        self.log_dir = param_dic['Log_dir'][0]
        self.iteration_gen = param_dic['Iteration_gen'][0][0] ; self.iteration_disc = param_dic['Iteration_disc'][0][0]
        self.lr_gen = param_dic['Learning_rate'][0][0]; self.lr_disc = param_dic['Learning_rate'][0][1]; self.beta=param_dic['Beta'][0]
        
    def read_img(self, files, size_y_data):
        imgs = []
        for img in range(self.OBSERVE_SIZE):
            x = tf.keras.preprocessing.image.load_img(self.data_set_path + files[img][0], 
                                                  color_mode='grayscale')
            x = tf.keras.preprocessing.image.img_to_array(x)
            x = x/255.0
            if len(imgs)==0:
                imgs = x
            else:
                imgs = tf.concat([imgs,x], axis=2)
        output = tf.keras.preprocessing.image.load_img(self.data_set_path + files[self.OBSERVE_SIZE][0],
                                                    color_mode='grayscale')
        output = tf.keras.preprocessing.image.img_to_array(output)  
        for y_img_num in range(1,size_y_data):
            y_output = tf.keras.preprocessing.image.load_img(self.data_set_path + files[self.OBSERVE_SIZE+y_img_num][0],
                                                         color_mode='grayscale')
            y_output = tf.keras.preprocessing.image.img_to_array(y_output)
            output = tf.concat([output, y_output], axis = 2)
        output = tf.convert_to_tensor(output) / 255.0
        return imgs, output
    
    def load_data_with_future_y_train(self, Y_TRAIN_SIZE):
        counter = 1 # Count the maximum size of y_train
        y_train_size = Y_TRAIN_SIZE
        for i in range(len(self.file_list)-self.OBSERVE_SIZE):
            counter = 1 # Count the maximum size of y_train
            if self.file_list[i][1]+self.OBSERVE_SIZE==self.file_list[i+self.OBSERVE_SIZE][1]:
                y_train_size = len(self.file_list)-i-self.OBSERVE_SIZE if (i+self.OBSERVE_SIZE+Y_TRAIN_SIZE>=len(self.file_list)) else Y_TRAIN_SIZE
                while (counter < y_train_size) and (self.file_list[i+counter+self.OBSERVE_SIZE-1][1]+1==self.file_list[i+counter+self.OBSERVE_SIZE][1]):
                    counter+=1
                x, y = self.read_img(self.file_list[i:i+self.OBSERVE_SIZE+counter], counter)
            self.x_train.append(x)
            self.y_train.append(y)
    
    def generate_images(self, sample_num, model = False, training = False, columns = 5, save = False):
        rows = self.OBSERVE_SIZE//columns +(self.OBSERVE_SIZE%columns > 0) +  1
        if model:
            prediction = self.generator(self.x_train[sample_num][tf.newaxis, ..., tf.newaxis], training= training)
            plt.subplot(rows, columns, (rows-1)*columns + columns//2 +1)
            plt.imshow(prediction[0,...,0], cmap = 'gray')
            plt.axis('off')
            plt.title('Predict')
        for i in range(self.OBSERVE_SIZE):
            axs = plt.subplot(rows, columns, i+1)
            axs.imshow(self.x_train[sample_num][:,:,i], cmap = 'gray')
            plt.title(i+1)
            plt.axis('off')
        plt.subplot(rows, columns, (rows-1)*columns + columns//2)
        plt.imshow(self.y_train[sample_num][...,1], cmap = 'gray')
        plt.title('Output')
        plt.axis('off')
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
            result.add(tf.keras.layes.Dropout(apply_dropout))
        
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
            self.upsample(256, self.kernel_size),
            self.upsample(128, self.kernel_size),
            self.upsample(52, self.kernel_size),
            self.upsample(36, self.kernel_size),
            self.upsample(28, self.kernel_size),
            self.upsample(10, self.kernel_size)
            ]
        last = tf.keras.layers.Conv3DTranspose(self.output_size, self.kernel_size, strides = (2,2,1), 
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
                shape = conc.shape
                if shape[3]!=1:
                    conc = tf.keras.layers.Reshape((shape[1], shape[2], 1, shape[3]*shape[4]))(conc)
                x = tf.keras.layers.Concatenate(axis = -1)([conc, x])
        x = last(x)
        
        self.generator = tf.keras.Model(inputs = inputs, outputs = x)
    
    def generator_loss(self, disc_gen_output, gen_output, target):
        gen_loss = self.loss_object(tf.ones_like(disc_gen_output),disc_gen_output)
        
        l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
        tot_loss = gen_loss +self.LAMBDA * l1_loss
        return tot_loss, gen_loss, l1_loss
        
    def create_discriminator(self):
        in_imgs = tf.keras.layers.Input(shape = [self.HEIGHT, self.WIDTH, self.OBSERVE_SIZE], name = 'input_imgs')
        tar_img = tf.keras.layers.Input(shape = [self.HEIGHT, self.WIDTH, self.output_size], name = 'target_img')
        
        conc = tf.keras.layers.Concatenate()([in_imgs, tar_img])
        
        down1 = tf.keras.layers.Conv2D(32, self.kernel_size, strides = 2, kernel_initializer = self.initializer,
                                       padding = 'same', use_bias=False)(conc)
        down2 = tf.keras.layers.Conv2D(128, self.kernel_size, strides = 2, kernel_initializer = self.initializer, 
                                       padding='same', use_bias = False)(down1)
        conv1 = tf.keras.layers.Conv2D(256, self.kernel_size, strides = 1, kernel_initializer = self.initializer,
                                       padding = 'valid', use_bias = False)(down2)
        conv2 = tf.keras.layers.Conv2D(512, self.kernel_size, strides = 1, kernel_initializer = self.initializer,
                                       padding = 'same', use_bias = False)(conv1)
        batch_norm = tf.keras.layers.BatchNormalization()(conv2)
        active_relu = tf.keras.layers.LeakyReLU(self.ALPHA)(batch_norm)
        last = tf.keras.layers.Conv2D(1, self.kernel_size, strides = 1, kernel_initializer = self.initializer,
                                      padding = 'same', use_bias = False)(active_relu)
        self.discriminator = tf.keras.Model(inputs = [in_imgs, tar_img], outputs = last)
    
    def discriminator_loss(self, disc_gen_output, disc_real_output):
        
        gen_loss = self.loss_object(tf.zeros_like(disc_gen_output),disc_gen_output)
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        
        return gen_loss + real_loss
    '''
    @tf.function
    def train_step(self, input_imgs, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            for train_gen in range(self.iteration_gen):
                gen_output  = self.generator(input_imgs, training = True)
                disc_gen_output = self.discriminator([input_imgs, gen_output[...,0]], training = True)
                gen_tot_loss, gen_loss, gen_l1_loss = self.generator_loss(disc_gen_output, gen_output[...,0], target)
                
                gradient_gen = gen_tape.gradient(gen_tot_loss, self.generator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
            for train_disc in range(self.iteration_disc):
                disc_real_output = self.discriminator([input_imgs, target], training = True)
                disc_loss = self.discriminator_loss(disc_gen_output, disc_real_output)
                
                gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                self.disc_optimizer.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))
        with self.summary_writer.as_default():
            tf.summary.scalar('Gen_total_loss', gen_tot_loss, step = epoch)
            tf.summary.scalar('Gen_loss', gen_loss, step = epoch)
            tf.summary.scalar('Gen_l1_loss', gen_l1_loss, step = epoch)
            tf.summary.scalar('Disc_loss', disc_loss, step = epoch)
    '''
    @tf.function
    def train_step(self, input_imgs, target, epoch):
        for train_gen in range(self.iteration_gen):
            with tf.GradientTape() as gen_tape:
                gen_output  = self.generator(input_imgs, training = True)
                disc_gen_output = self.discriminator([input_imgs, gen_output[...,0]], training = True)
                gen_tot_loss, gen_loss, gen_l1_loss = self.generator_loss(disc_gen_output, gen_output[...,0], target)
                
                gradient_gen = gen_tape.gradient(gen_tot_loss, self.generator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
           
        for train_disc in range(self.iteration_disc):
            with tf.GradientTape() as disc_tape:
                disc_real_output = self.discriminator([input_imgs, target], training = True)
                disc_loss = self.discriminator_loss(disc_gen_output, disc_real_output)
                    
                gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                self.disc_optimizer.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))
        
        with self.summary_writer.as_default():
            tf.summary.scalar('Gen_total_loss', gen_tot_loss, step = epoch)
            tf.summary.scalar('Gen_loss', gen_loss, step = epoch)
            tf.summary.scalar('Gen_l1_loss', gen_l1_loss, step = epoch)
            tf.summary.scalar('Disc_loss', disc_loss, step = epoch)
    
    def fit(self, epochs, model_name):
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + 'fit/'+model_name + datetime.datetime.now().strftime(' %d.%m.%y--%H:%M:%S'))
        if not os.path.exists(model_name+'/figs'):
            os.makedirs(model_name+'/figs')
        for epoch in range(epochs):
            start = time.time()
            print('Epoch:', epoch+1)
            with open('log.txt','a') as f:
                print('Epoch:', epoch+1, file=f)
            for n, (input_imgs, target_img) in enumerate(zip(self.x_train, self.y_train)):
                for i in range(tf.shape(target_img)[2]):
                    self.train_step(input_imgs[tf.newaxis,...], target_img[tf.newaxis,:,:,i], epoch+1)
                    gen_img = self.generator(input_imgs[tf.newaxis,...])
                    input_imgs = tf.concat([input_imgs, gen_img[0,...,0]], axis = -1)
                    input_imgs = input_imgs[:,:,:self.OBSERVE_SIZE]
                print('.',end='')
                if (n+1)%100==0:
                    print('\n')
            if (epoch+1)%20==0:
                self.generate_images(0, model =True, save = '{}/figs/epoch-{}.png'.format(model_name,epoch+1))
                self.generator.save(model_name+'/generator')
                self.discriminator.save(model_name + '/discriminator')
            print('\nTime taken to epoch: {} in sec {}\n'.format(epoch, time.time()-start))
        self.generator.save(model_name+'/generator')
        self.discriminator.save(model_name + '/discriminator')
          

    def print_model(self):
        tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi = 96, to_file='Generator.png')
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi = 96, to_file='Discriminator.png')

# model = Three_d_conv_model('/home/lab/orel_ws/project/simulation_ws/data_set/','3D_conv_try',OBSERVE_SIZE=3,
#                             Y_TRAIN_SIZE=2,LR_GEN=2e-5,concate=False)
model_2 = Three_d_conv_model('/home/lab/orel_ws/project/simulation_ws/data_set/','3D_conv_try',load_model=True)
# model = Three_d_conv_model(data_set_path,Y_TRAIN_SIZE=1)
# model.create_generator()
# model.create_discriminator()
# model.generate_images(1,model = True)



