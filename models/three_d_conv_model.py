#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
import datetime
import copy
from scipy.io import savemat, loadmat
from states_and_predictions import add_border


class Three_d_conv_model():
    def __init__(self, model_name, data_set_path ='', load_model = False, OBSERVE_SIZE = 5,
                 Y_TRAIN_SIZE = 1, HEIGHT = 128, WIDTH = 128, ALPHA = 0.2, kernel_size = 3,
                 LAMBDA = 20, LR_GEN = 15e-5, BETA_1_GEN =0.5, BETA_2_GEN =.999, LR_DISC= 2e-4, BETA_1_DISC =0.5,
                 BETA_2_DISC =.999, prediction_gap = 0, concate = True):

        self.file_num = lambda x: int(x[x.index('--')+2:x.index('.jpg')])
        self.initializer = tf.random_normal_initializer(0.,0.02) # Var= 0.02
        self.model_name = model_name
        if not load_model:
            self.OBSERVE_SIZE = OBSERVE_SIZE ; self.Y_TRAIN_SIZE = Y_TRAIN_SIZE
            self.height = HEIGHT; self.width = WIDTH 
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
            self.load_parm(model_name)
            if data_set_path!='': self.data_set_path = data_set_path
            try:
                self.generator = tf.keras.models.load_model(model_name + '/generator')
                self.discriminator = tf.keras.models.load_model(model_name + '/discriminator')
            except OSError:
                print("Generator and discriminator didn't found, and will initialize.")
                self.generator = self.create_generator()
                self.discriminator = self.create_discriminator()

        self.load_reff_disc(model_name)
        if not self.discriminator_reff:
            print("Reff disctiminator wasn't found. Use fit method to train one")
        gap = 1
        if 'armadillo' in self.data_set_path: gap = 10  
        self.file_list = [[file, self.file_num(file)] for file in os.listdir(self.data_set_path) 
                  if file.endswith('.jpg') and self.file_num(file)%gap==0] # For armadillo data set]
        self.file_list.sort(key = lambda x:x[1])

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.initializer = tf.random_normal_initializer(0.,0.02) # Var =0.02

        self.load_data()
        self.gen_optimizer = tf.keras.optimizers.Adam(self.generator_learning_rate, beta_1 =self.beta[0], beta_2 = self.beta[1])
        self.disc_optimizer = tf.keras.optimizers.Adam(self.discriminator_learning_rate, beta_1= self.beta[2], beta_2 = self.beta[3])

    def save_pram(self,model_name):
        dic = {'OBSERVE_SIZE': self.OBSERVE_SIZE,'Y_train_size':self.Y_TRAIN_SIZE, 'Height':self.height,'Width':self.width, 'Alpha':self.ALPHA,
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
        self.height = param_dic['Height'][0][0]; self.width = param_dic['Width'][0][0]
        self.ALPHA = param_dic['Alpha'][0][0] # Alpha for leakyReLU
        self.kernel_size = (np.ones(3,)*param_dic['Kernel_size'][0][0]).astype(np.uint8)
        self.LAMBDA = param_dic['Lambda'][0][0]
        self.concate = param_dic['Concate'][0][0]
        self.data_set_path = str(param_dic['Data_set_path'][0])
        self.learning_rates = param_dic['Learning_rates'][:,0].reshape((2,1)); self.beta=param_dic['Beta'][0]
        self.prediction_gap = param_dic['Prediction_gap'][0][0]

    def load_reff_disc(self, model_name):
        try:
            self.discriminator_reff = tf.keras.models.load_model(model_name+'/discriminator_reff')
        except IOError:
            self.discriminator_reff = False

    def read_img(self, img_path):
        x = tf.keras.preprocessing.image.load_img(img_path, 
                                                  color_mode='grayscale')
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = x/127.5-1
        return x

    def load_data(self):
        self.train_sequence = self.read_img(self.data_set_path + self.file_list[0][0])
        for img_name in self.file_list[1:]:
            self.train_sequence = np.concatenate((self.train_sequence,
                                                  self.read_img(self.data_set_path + img_name[0])),
                                                 axis=2)
        self.data_set_size = self.train_sequence.shape[2]

    def generate_images(self, inx, model = False, training = False, columns = 5, save = False):
        columns = self.OBSERVE_SIZE
        rows = 3 # input, target, prediction
        prediction = self.generator(self.train_sequence[tf.newaxis,:,:,inx:inx + self.OBSERVE_SIZE, tf.newaxis], training= training)[0,...,0]
        for i in range(self.OBSERVE_SIZE):
            axs = plt.subplot(rows, columns, i+1)
            axs.imshow(self.train_sequence[...,inx+i], cmap = 'gray', vmin = -1, vmax = 1)
            # plt.title(i+1)
            plt.axis('off')
            plt.subplot(rows, columns, columns + i+1)
            plt.imshow(self.train_sequence[...,inx+self.OBSERVE_SIZE+i+1], cmap = 'gray', vmin = -1, vmax = 1)
            # plt.title('Output {}'.format(str(i)))
            plt.axis('off')
            if model:
                plt.subplot(rows, columns, 2*columns + i+1)
                plt.imshow(prediction[:,:,i], cmap = 'gray', vmin = -1, vmax = 1)
                plt.axis('off')
                # plt.title('Predict')
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
        inputs = tf.keras.layers.Input(shape = (self.height, self.width, self.OBSERVE_SIZE, 1))
        downing = [
            self.downsample(16, self.kernel_size),
            self.downsample(32, self.kernel_size),
            self.downsample(48, self.kernel_size),
            self.downsample(64, self.kernel_size),
            self.downsample(128, self.kernel_size),
            self.downsample(256, self.kernel_size),
            self.downsample(512, self.kernel_size)
            ]
        upping = [
            self.upsample(256, self.kernel_size, apply_dropout = 0.5),
            self.upsample(128, self.kernel_size, apply_dropout = 0.5),
            self.upsample(64, self.kernel_size, apply_dropout = 0.5),
            self.upsample(48, self.kernel_size),
            self.upsample(32, self.kernel_size),
            self.upsample(16, self.kernel_size)
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
        
        in_imgs = tf.keras.layers.Input(shape = [self.height, self.width, self.OBSERVE_SIZE, 1], name = 'input_imgs')
        tar_img = tf.keras.layers.Input(shape = [self.height, self.width, self.OBSERVE_SIZE, 1], name = 'target_imgs')
        
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
        
        gen_loss = self.loss_object(tf.zeros_like(disc_gen_output), disc_gen_output)
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        
        return gen_loss + real_loss
    
    def generator_learning_rate(self):
        if self.epoch+1<50 or self.learning_rates[0,-1]<2e-6:
            return self.learning_rates[0,-1]
        
        diff_loss_avg = np.diff(self.losses[3,-20:]).mean()
        if diff_loss_avg > 0:
            return self.learning_rates[0,-1]*0.92
        
        else:
            return self.learning_rates[0,-1]*0.965
        
    def discriminator_learning_rate(self):
        if self.epoch+1<50 or self.learning_rates[1,-1]<2e-6:
            return self.learning_rates[1,-1]
        
        diff_loss_avg = np.diff(self.losses[3,-20:]).mean()
        if diff_loss_avg > 0:
            return self.learning_rates[1,-1]*0.985
        
        else:
            return self.learning_rates[1,-1]*0.935
        
    def train_step(self, input_imgs, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            gen_output  = self.generator(input_imgs, training = True)
            disc_gen_output = self.discriminator([input_imgs, gen_output], training = True)
            gen_tot_loss, gen_loss, gen_l1_loss = self.generator_loss(disc_gen_output, gen_output, target)
            
            gradient_gen = gen_tape.gradient(gen_tot_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
       
            disc_real_output = self.discriminator([input_imgs, target], training = True)
            disc_loss = self.discriminator_loss(disc_gen_output, disc_real_output)
                
            gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))
            
            self.losses_val = np.append(self.losses_val,[[gen_tot_loss.numpy()],[gen_loss.numpy()],
                                                         [gen_l1_loss.numpy()],[disc_loss.numpy()]], axis = 1)
                   
    def sample_imgs(self, epoch, model_name):
        self.generate_images(1, model = self.generator, save = '{}/figs/epoch--{}.png'.format(model_name,epoch+1))
        inx = tf.random.uniform([1],0,self.data_set_size - 2*self.OBSERVE_SIZE-self.prediction_gap, dtype = tf.dtypes.int32).numpy()[0]
        self.generate_images(inx, model = self.generator, save = '{}/figs/random/epoch--{}_inx-{}.png'.format(model_name,epoch+1,inx))
      
    def fit(self, epochs, model_name, disc_reff = False):
        self.losses = np.zeros((5,0))
        self.losses_val = np.zeros((4,0))
        self.learning_rates = np.array(self.learning_rates[:,0]).reshape((2,1))
        if disc_reff:
            self.load_reff_disc(model_name)
            if not self.discriminator_reff: # not
                print("Unable to load Reff discriminator. Start training one...")
            else:
                del self.gen_optimizer
                del self.disc_optimizer
                try:
                    del self.generator
                    del self.discriminator
                except AttributeError:
                    pass
                self.gen_optimizer = tf.keras.optimizers.Adam(self.generator_learning_rate, beta_1 =self.beta[0], beta_2 = self.beta[1])
                self.disc_optimizer = tf.keras.optimizers.Adam(self.discriminator_learning_rate, beta_1= self.beta[2], beta_2 = self.beta[3])
                self.create_generator()
                self.create_discriminator()
                
        if not os.path.exists(model_name+'/figs/random'):
            os.makedirs(model_name+'/figs/random')
        with open(model_name +'/read me.txt','a') as f:
               f.write('\n {} {}, Epochs:{}, Y_train (recursive):{}, Prediction gap:{}, Input&output size: {}.'
                       .format(model_name, datetime.datetime.now().strftime('%Y.%m.%d--%H:%M:%S'), epochs,
                               self.Y_TRAIN_SIZE, self.prediction_gap, self.OBSERVE_SIZE))
               if not self.discriminator_reff:
                   f.write(' Reff training.')
               f.close()
        self.sample_imgs(-1, model_name)
        for epoch in range(epochs):
            start = time.time()
            self.epoch = epoch
            reff_loss_avg = 0
            print('Epoch:'+ str(epoch+1))
            
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
                        reff_loss = self.discriminator_loss(disc_gen.numpy(), disc_real.numpy())
                        reff_loss_avg += reff_loss/(self.data_set_size-2*self.OBSERVE_SIZE-self.prediction_gap)
                    # input_seq = tf.concat([input_seq, gen_img[0,...,0]], axis = 2)
                    # input_seq = input_seq[:,:,1:]
                print('.', end = '')
                if (img_inx+1)%100==0:
                    print('\n')
            np.append(self.losses_val.mean(axis=1),[reff_loss_avg], axis=0)                   
            self.losses = np.append(self.losses, np.append(self.losses_val.mean(axis=1),[reff_loss_avg], axis=0).reshape((5,1)), axis = 1)
            if (epoch+1)%200==0:
                if not self.discriminator_reff:
                    self.generator.save(model_name+'/generator_0')
                    self.discriminator.save(model_name + '/discriminator_reff')
                else:
                    self.generator.save(model_name+'/generator')
                    self.discriminator.save(model_name + '/discriminator')
                    
            self.sample_imgs(epoch, model_name)
            lr_rates = np.array((self.gen_optimizer.get_config()['learning_rate'],
                                  self.disc_optimizer.get_config()['learning_rate'])).reshape(2,1)
            self.learning_rates = np.append(self.learning_rates, lr_rates, axis = 1)
            print('\nTime taken to epoch: {} in sec {}\n'.format(epoch+1, time.time()-start))
        if self.discriminator_reff:
            self.generator.save(model_name+'/generator')
            self.discriminator.save(model_name + '/discriminator')
        else: 
            self.generator.save(model_name+'/generator_0')
            self.discriminator.save(model_name + '/discriminator_reff')
            os.rename(model_name + '/figs',model_name + '/figs_reff')
        
        self.save_losses(model_name)
    
    def save_losses(self, model_name):
        losses_dic = {'Gen_total_loss': self.losses[0,:], 'Gen_loss': self.losses[1,:],
                     'Gen_l1_loss': self.losses[2,:],'Disc_loss': self.losses[3,:],
                     'Reff_disc_loss': self.losses[4,:]}
        lr_dic = {'gen_lr': self.learning_rates[0,:], 'disc_lr': self.learning_rates[1,:]}
        savemat('{}/losses-{}.mat'.format(model_name, datetime.datetime.now().strftime('%m.%d--%H:%M:%S')),losses_dic)
        savemat('{}/lr_rates-{}.mat'.format(model_name, datetime.datetime.now().strftime('%m.%d--%H:%M:%S')),lr_dic)

    def print_model(self):
        tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi = 96, to_file='{}/Generator.png'.format(self.model_name))
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi = 96, to_file='{}/Discriminator.png'.format(self.model_name))
    
    def create_seq_img(self, input_data, target):
        assert self.generator, "Generator isn't exict, use method create_generator() \
                                and train it, or use trained model."
        gen_output = self.generator(input_data[tf.newaxis,...,tf.newaxis])[0,...,0]
        input_imgs = []
        target_imgs = []; gen_imgs = []
        for i in range(self.OBSERVE_SIZE):
            input_imgs.append(add_border(input_data[...,i]))
            target_imgs.append(add_border(target[...,i]))
            gen_imgs.append(add_border(gen_output[...,i]))
        height = input_imgs[0].shape[0]
        input_imgs = np.transpose(input_imgs, (1,2,0)).reshape((height,-1), order = 'F')
        target_imgs = np.transpose(target_imgs, (1,2,0)).reshape((height,-1), order = 'F')
        gen_imgs = np.transpose(gen_imgs, (1,2,0)).reshape((height,-1), order = 'F')
        gen_imgs = np.concatenate((np.ones((height, gen_imgs.shape[1])),gen_imgs), axis = -1)
        input_imgs = np.concatenate((input_imgs, target_imgs),axis = -1).reshape(input_imgs.shape[0],-1)
        return np.concatenate((input_imgs, gen_imgs),axis = 0 )
            
    def create_test_set(self, start_inx, end_inx, test_path):
        gap = 1
        if 'armadillo' in self.data_set_path:
            start_inx *=10; end_inx *=10; gap = 10 # For armadillo data-set
            # Create the test_set with the update data path
        file_list = [[file, self.file_num(file)] for file in os.listdir(test_path) 
                     if file.endswith('.jpg') and self.file_num(file)>=start_inx and
                     self.file_num(file)<end_inx and self.file_num(file)%gap==0] # - Armadillo Generate file list with the range.
        file_list.sort(key= lambda x:x[1])
        end_inx = file_list[-1][1]
        test_set = self.read_img(test_path + file_list[0][0])
        self.test_files = file_list
        for img_name in file_list[1:]:
            test_set = np.concatenate((test_set, self.read_img(test_path + img_name[0])), axis=2)
        return test_set
        
        
    def model_validation(self, start_inx = 0, end_inx = -1, test_path = False, pause_time =0.25):
        if end_inx==-1:
            end_inx = start_inx+self.OBSERVE_SIZE*5 # default is 5 times the observe size
        if not test_path:
            if end_inx>self.data_set_size: end_inx=self.data_set_size
            test_set = self.train_sequence[:,:,start_inx:end_inx]
        else:
            test_set = self.create_test_set(start_inx, end_inx, test_path)
        for inx in range(np.shape(test_set)[2]):
            input_data = test_set[:,:,inx:inx + self.OBSERVE_SIZE]
            target = test_set[:,:,inx +self.OBSERVE_SIZE:inx +2*self.OBSERVE_SIZE]
            if target.shape[2]<self.OBSERVE_SIZE:
                break
            img = self.create_seq_img(input_data, target)
            normal = np.zeros(img.shape)
            normal = cv2.normalize(img, normal, 0, 1, cv2.NORM_MINMAX)
            self.normal = normal
            
            # self.img = img
            cv2.imshow('display', normal)
            k = cv2.waitKey(1)
            if k ==27 or k== ord('q'): 
                break
            elif k==ord('a'):
                time.sleep(1.5)
            elif k==ord('s'):
                mpimg.imsave("{}/sample-{} model-{}.png".format(self.model_name, str(start_inx+inx+1), self.model_name), normal, cmap = 'gray')
            time.sleep(pause_time)
            
        
if __name__ == '__main__':     
    model_name = 'ARM-3D_conv'
    model = Three_d_conv_model(model_name,
                            load_model=True)


    # model = Three_d_conv_model('/home/lab/orel_ws/project/data_set_armadillo/3/', 
    #                             model_name, OBSERVE_SIZE = 5, load_model = False)
    
    # model.model_validation()
    
    # model.print_model()
    # model.fit(150, model_name, disc_reff=False)
    # model.fit(150, model_name, disc_reff=True)
    
    # model.model_validation(0,1000, test_path='/home/lab/orel_ws/project/data_set/test/')
    
    model.model_validation(0,350,test_path='/home/lab/orel_ws/project/data_set_armadillo/2/') 




