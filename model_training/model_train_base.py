#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import copy
from scipy.io import savemat


data_set_path = '/home/lab/orel_ws/project/src/simulation_ws/data_set/'
# train_dataset = tf.data.Dataset.list_files(data_set_path + '/*.jpg')
file_num = lambda x: int(x[x.index('--')+2:x.index('.jpg')])

file_list = [[file, file_num(file)] for file in os.listdir(data_set_path)
             if file.endswith('.jpg')]
file_list.sort(key = lambda x:x[1])
file_list_test =[]
file_list_train = []
DATA_SET_SIZE = len(file_list)
BATCH_SIZE = 1
HEIGHT, WIDTH = 128, 128
EPOCHS = 10
VAR = 0.02 # Variance of initialize kernels.
ALPHA = 0.2 # Alpha for leakyReLU.
DROP_RATE = 0.5 # Dropout rate for upsample.
OBSERVE_SIZE = 5 # How many img to observe.
Y_TRAIN_SIZE = 1 # How many img to learn recursive.
GAP_PREDICT = 3 # The gap between the last observe and the predict.
OUTPUT_SIZE = 1
LAMBDA = 20 # determine the weight of l1 in Loss function (generator) LAMBDA = 0 -> cancel l1_loss
loss_object  = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LR_GEN =2e-5; BETA_1_GEN =0.5; BETA_2_GEN =.5
LR_DISC =2e-4; BETA_1_DISC =0.5; BETA_2_DISC =.5
ITERATION_GEN = 1 ; ITERATION_DISC = 1
model_name = 'cGAN_5pic_1y_train_1.7'
losses_val = np.zeros((4,0))
losses_avg = np.zeros((5,0)) # [Gen_total_loss, Gen_loss, Gen_l1_loss, Disc_loss, Reff_disc_loss]
learning_rates = np.zeros((2,0))

if not os.path.exists(model_name):
    os.makedirs(model_name)

def read_img(img_name):
    imgs = []

    x = tf.keras.preprocessing.image.load_img(data_set_path + img_name,
                                              color_mode='grayscale')
    x = tf.keras.preprocessing.image.img_to_array(x)
    x = x/127.5 - 1
    
    #     if len(imgs)==0:
    #         imgs = x
    #     else:
    #         imgs = tf.concat([imgs,x], axis=2)
    # output = tf.keras.preprocessing.image.load_img(data_set_path + files[OBSERVE_SIZE][0],
    #                                                 color_mode='grayscale')
    # output = tf.keras.preprocessing.image.img_to_array(output)
    # for y_img_num in range(1,size_y_data):
    #     y_output = tf.keras.preprocessing.image.load_img(data_set_path + files[OBSERVE_SIZE+y_img_num][0],
    #                                                      color_mode='grayscale')
    #     y_output = tf.keras.preprocessing.image.img_to_array(y_output)
    #     output = tf.concat([output, y_output], axis = 2)
    # output = tf.convert_to_tensor(output) / 127.5 - 1
    return x
    
    
def load_data():
    train_sequence =read_img(file_list[0][0]) 
    for img_name in file_list[1:]:
        train_sequence = np.concatenate((train_sequence , read_img(img_name[0])), axis=2)

    
    return train_sequence
'''
def load_data():
    x_train =[]
    y_train =[]
    for i in range(BUFFER_SIZE-OBSERVE_SIZE):
        if file_list[i][1]+OBSERVE_SIZE==file_list[i+OBSERVE_SIZE][1]:
            x, y = read_img(file_list[i:i+OBSERVE_SIZE+1],1)
            x_train.append(x)
            y_train.append(y)
    return x_train, x_test, y_train, y_test
'''

        

def generate_image(inx, model = False, training = False, save =False):
    if save: 
        path = os.path.split(save)[0]
        if not os.path.exists(path):
            os.makedirs(path)
    if model:
        prediction = model(train_sequence[tf.newaxis,:,:,inx:inx+OBSERVE_SIZE], training= training)
        plt.subplot(3,5,14)
        plt.imshow(prediction[0,...], cmap = 'gray', vmin = -1, vmax = 1)
        plt.axis('off')
        plt.title('Predict')
    for i in range(OBSERVE_SIZE):
        axs = plt.subplot(3,5,i+1)
        axs.imshow(train_sequence[:,:,inx+i], cmap = 'gray',  vmin = -1, vmax = 1)
        plt.title(i+1)
        plt.axis('off')
    plt.subplot(3,5,13)
    plt.imshow(train_sequence[:,:,inx+OBSERVE_SIZE+GAP_PREDICT+1], cmap = 'gray',  vmin = -1, vmax = 1)
    plt.title('Output')
    plt.axis('off')
    if not save:
        plt.show()
    else:
        plt.savefig(save)
    
# ----------------- Random generate imgs -------------------------    
#inx = tf.random.uniform([1],0,len(train_sequence)-OBSERVE_SIZE-1,dtype = tf.dtypes.int32).numpy()[0]
#generate_image(inx)

# ----------------------------------------------------------------

def downsample(filters, size, apply_batchnorm = True):
    initializer = tf.random_normal_initializer(0.,VAR)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,size,
                                      strides=2, padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU(ALPHA))
    
    return result

def upsample(filters, size, apply_batchnorm = False, apply_dropout = False):
    initializer = tf.random_normal_initializer(0.,VAR)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters,size,
                                      strides=2, padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(apply_dropout))
    
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[HEIGHT, WIDTH,OBSERVE_SIZE])
    
    downing = [ 
        downsample(32, 4, apply_batchnorm=False), # (bs, 64,64,32)
        downsample(64, 4), # (bs, 32,32,64)
        downsample(128, 4), # (bs, 16,16,128)
        downsample(256, 4), # (bs, 8,8,256)
        downsample(512, 4), # (bs, 4,4,512)
        downsample(512, 4), # (bs, 2,2,512)
        downsample(512, 4), # (bs, 1,1,512)
        ]
    
    upping = [
        upsample(512, 4, apply_dropout = DROP_RATE), # (bs, 2,2,512)
        upsample(512, 4, apply_dropout = DROP_RATE), # (bs, 4,4,512)
        upsample(256, 4, apply_dropout = DROP_RATE), # (bs, 8,8,256)
        upsample(128, 4), # (bs, 16,16,128)
        upsample(64, 4), # (bs, 32,32,64)
        upsample(32,4), # (bs, 64,64,32)
        ]
    initializer = tf.random_normal_initializer(0.,VAR)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_SIZE, 4, strides = 2,
                                           padding= 'same', activation = 'tanh',
                                           kernel_initializer = initializer)
    
    x = inputs
    connections = []
    for down in downing:
        x = down(x)
        connections.append(x)
    connections = reversed(connections[:-1])
    
    for up, conc in zip(upping, connections):
        x = up (x)
        x = tf.keras.layers.Concatenate()([conc , x])
    x = last(x)
    
    return tf.keras.Model(inputs = inputs, outputs = x)

def generator_loss(disc_generated_output, generated_output, target):
    gen_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(generated_output - target))
    tot_loss = gen_loss + LAMBDA*l1_loss 
    return tot_loss, gen_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0.,VAR)
    
    in_gen = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, OBSERVE_SIZE],
                                   name = 'input_imgs') # (bs, 128,128,10)
    in_tar = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1], 
                                   name ='target_img') # (bs, 128,128,1)
        
    conc = tf.keras.layers.Concatenate(axis=-1) ([in_gen, in_tar]) # (bs, 128,128,11)
    down1 = downsample(32, 4, apply_batchnorm=False) (conc) # (bs, 64,64,32)
    down2 = downsample(128, 4) (down1) # (bs, 32,32,128)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D() (down2) # (bs, 34,34,128)
    conv = tf.keras.layers.Conv2D(256, 4, strides = 1,
                                  kernel_initializer = initializer,
                                  padding = 'valid', use_bias = False) (zero_pad1) # (bs, 31,31,256)
    batch_norm = tf.keras.layers.BatchNormalization() (conv) # (bs, 31,31,256)
    active_relu = tf.keras.layers.LeakyReLU(ALPHA) (batch_norm)  # (bs, 31,31,256)
    zero_pad2 = tf.keras.layers.ZeroPadding2D() (active_relu) # (active_relu)  # (bs, 33,33,256)
    conv2 = tf.keras.layers.Conv2D(512, 4, strides = 1,
                                  kernel_initializer = initializer,
                                  padding = 'valid', use_bias = False) (zero_pad2) # (bs, 30,30,512)
    last = tf.keras.layers.Conv2D(1, 4, strides = 1,
                                  kernel_initializer = initializer,
                                  padding = 'same', use_bias = False) (conv2) # (bs, 30,30,1)
    
    return tf.keras.Model(inputs = [in_gen, in_tar], outputs = last)  

def discriminator_loss(disc_generated_output, disc_real_outpput):
        
    real_loss = loss_object(tf.ones_like(disc_real_outpput), disc_real_outpput)
    gen_loss = loss_object(tf.zeros_like(disc_real_outpput), disc_generated_output)
    
    return real_loss ,gen_loss
    
    


# ----------- Print exmaple of input and predict -------------- 
# # for _ in range(5):
#     inx = tf.random.uniform([1],0,len(x_test),dtype = tf.dtypes.int32).numpy()[0]
#     plt.figure()
#     generate_image(x_test[inx], y_test[inx], model = generator )
#     print(file_list_test[inx])

# inx = tf.random.uniform([1],0,len(x_train),dtype = tf.dtypes.int32).numpy()[0]
# plt.figure()
# generate_image(x_train[inx], y_train[inx], model = generator )
# print(file_list_train[inx])
# # disc_out = discriminator([x_train[0][tf.newaxis, ...], tf.expand_dims(y_train[0],axis=0)] ,training = False)
# # plt.figure()
# # plt.imshow(disc_out[0,...], cmap = 'RdBu_r')

# ------------------------------------------------------------



def train_step(input_imgs, target, epoch, step):
    
    global losses_val, disc_gen_loss
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_imgs[tf.newaxis,...], training = True)
        disc_gen_output = discriminator([input_imgs[tf.newaxis,...], gen_output], training = True)
        gen_tot_loss, gen_loss, gen_l1_loss = generator_loss(disc_gen_output,
                                                             gen_output[0,:,:,0],
                                                             target)
        gradient_gen = gen_tape.gradient(gen_tot_loss, 
                                         generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradient_gen,
                                                generator.trainable_variables))
        
        disc_real_output = discriminator([input_imgs[tf.newaxis,...], target[tf.newaxis,...,tf.newaxis]],
                                         training = True)
        disc_loss, disc_gen_loss = discriminator_loss(disc_gen_output, disc_real_output)
            
        gradient_disc = disc_tape.gradient(disc_loss + disc_gen_loss,
                                           discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradient_disc,
                                                    discriminator.trainable_variables))
            
    
    #disc_gen_loss = np.append(disc_gen_loss, disc_gen_loss.numpy())
    losses_val = np.append(losses_val,[[gen_tot_loss.numpy()],[gen_loss.numpy()],[gen_l1_loss.numpy()]
                                        ,[(disc_loss+disc_gen_loss).numpy()]],axis = 1)


def sample_imgs(epoch):
    generate_image(50, model = generator, save = '{}/figs/epoch--{}.png'.format(model_name,epoch+1))
    inx = tf.random.uniform([1],0,train_sequence.shape[2]-OBSERVE_SIZE-1,dtype = tf.dtypes.int32).numpy()[0]
    generate_image(inx, model = generator, save = '{}/figs/random/epoch--{}_inx-{}.png'.format(model_name,epoch+1,inx))


def learning_rate_gen():
    if epoch+1<50 or epoch%10!=0:
        #LR_GEN = generator_optimizer.get_config()['learning_rate']
        return LR_GEN
    global losses_avg
    diff_loss_avg = np.diff(losses_avg[3,-10:]).mean()
    if diff_loss_avg> 0:
        return LR_GEN*0.9
    else:
        return LR_GEN*1.5
    
def learning_rate_disc():
    if epoch+1<50 or epoch%10!=0:
        #LR_DISC = discriminator_optimizer.get_config()['learning_rate']
        return LR_DISC
    global losses_avg
    diff_loss_avg = np.diff(losses_avg[3,-10:]).mean()
    if diff_loss_avg> 0:
        return LR_DISC*1.5
    else:
        return LR_DISC*0.9
def show_img(images,target):
    for i in range(images.shape[-1]):
        axs = plt.subplot(3,5,i+1)
        axs.imshow(images[:,:,i], cmap = 'gray',  vmin = -1, vmax = 1)
        plt.title(i+1)
        plt.axis('off')
    plt.subplot(3,5,13)
    plt.imshow(target, cmap = 'gray',  vmin = -1, vmax = 1)
    plt.title('Output')
    plt.axis('off')
    plt.show()
    
    
def fit(train_sequence, epochs = EPOCHS, step = 0, model_name= 'generic_model'):
    with open(model_name +'/read me.txt','a') as file: 
        file.write('\n' + model_name + 
                   datetime.datetime.now().strftime('%Y.%m.%d--%H:%M:%S') +' Epochs:'+str(epochs)
                   + ' Y_train is:' + str(Y_TRAIN_SIZE) + ' Prediction gap:' + str(GAP_PREDICT))
        if not discriminator_reff:
            file.write(', Reff')
        file.close()
    global losses_avg, epoch, disc_gen_loss, disc_gen_loss_avg, learning_rates, losses_val
    # disc_gen_loss = np.zeros((1,0))
    # disc_gen_loss_avg = np.zeros((1,0))
    
    sample_imgs(-1) 
    for epoch in range(epochs):
        start = time.time()
        reff_loss = 0
        losses_val =  np.zeros((4,0))
        print('Epoch:', epoch+1)
        for img_inx in range(DATA_SET_SIZE-OBSERVE_SIZE-1-GAP_PREDICT):
            input_seq = copy.copy(train_sequence[:,:,img_inx:img_inx+OBSERVE_SIZE])
            # train_step(train_sequence[:,:,img_inx:img_inx+OBSERVE_SIZE], train_sequence[:,:,img_inx+OBSERVE_SIZE+1], 
                               # epoch, step)
            for rec in range(Y_TRAIN_SIZE):
                if (img_inx+OBSERVE_SIZE+rec+1+GAP_PREDICT)<DATA_SET_SIZE:
                    train_step(input_seq, train_sequence[:,:,img_inx+OBSERVE_SIZE+rec+GAP_PREDICT+1], 
                                epoch, step)
                    gen_img = generator(input_seq[tf.newaxis,...])     
                    input_seq = tf.concat([input_seq, gen_img[0]], axis=-1)
                    input_seq = input_seq[:,:,1:]
                    
            if discriminator_reff: 
                input_seq = train_sequence[:,:,img_inx:img_inx+OBSERVE_SIZE]
                gen_output = generator(input_seq[tf.newaxis,...],
                                       training = False)
                disc_gen = discriminator_reff([input_seq[tf.newaxis,...], gen_output], training = False)
                disc_real = discriminator_reff([input_seq[tf.newaxis,...], train_sequence[tf.newaxis,:,:,img_inx+OBSERVE_SIZE+GAP_PREDICT+1]]
                                               , training = False)
                reff_real_loss, reff_gen_loss = discriminator_loss(disc_gen, disc_real)
                reff_loss += reff_gen_loss/(DATA_SET_SIZE-OBSERVE_SIZE-1)
            
            print('.' ,end='')
            if (img_inx+1)%100==0:
                    print('\n')
        LR_DISC = discriminator_optimizer.get_config()['learning_rate']
        LR_GEN = generator_optimizer.get_config()['learning_rate']
        learning_rate = np.array((LR_GEN, LR_DISC)).reshape(2,1)
        learning_rates = np.append(learning_rates,learning_rate, axis= 1)
        # disc_gen_loss_avg = np.append(disc_gen_loss_avg, disc_gen_loss.mean())
        losses_avg = np.append(losses_avg, np.append(losses_val.mean(axis =1),reff_loss).reshape(5,1), axis =1)         
        if (epoch+1)%200==0:
            if not discriminator_reff:
                generator.save(model_name+'/generator_0')
                discriminator.save(model_name+'/discriminator_reff')
            else:    
                generator.save(model_name+'/generator')
                discriminator.save(model_name+'/discriminator')
        sample_imgs(epoch)
        print('\nTime taken to epoch: {} in sec {}\n'.format(epoch+1, time.time()- start))
        

if __name__ =='__main__':
    train_sequence = load_data()
    # train_sequence = train_sequence[:,:,:100]
    # DATA_SET_SIZE = 100
    
    try: 
        generator = tf.keras.models.load_model(model_name+'/generator')
    except OSError:
        generator = Generator()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate_gen, beta_1=BETA_1_GEN, beta_2=BETA_2_GEN)

    tf.keras.utils.plot_model(generator, show_shapes=True, 
                              dpi = 96, to_file=model_name+'/Generator.png')    
    try: 
        discriminator = tf.keras.models.load_model(model_name+'/discriminator')
    except OSError:
        discriminator = Discriminator()

    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_disc, beta_1=BETA_1_DISC, beta_2=BETA_2_DISC)

    try:
        discriminator_reff = tf.keras.models.load_model(model_name+'/discriminator_reff')
    except OSError:
        discriminator_reff = False
    
    tf.keras.utils.plot_model(discriminator, show_shapes = True,
                              dpi = 96, to_file = model_name + '/Discriminator.png')  
    fit(train_sequence, epochs= 150, model_name=model_name)
    if not discriminator_reff: # if not discriminator_reff:
        generator.save(model_name+'/generator_0')
        discriminator.save(model_name+'/discriminator_reff')
        os.rename(model_name + '/figs',model_name + '/figs_reff')
    else:    
        generator.save(model_name+'/generator')
        discriminator.save(model_name+'/discriminator' )
    
    lr_rates =  {'gen_lr': learning_rates[0,:], 'disc_lr': learning_rates[1,:]}
    dic = {'Gen_total_loss': losses_avg[0,:], 'Gen_loss': losses_avg[1,:], 'Gen_l1_loss':losses_avg[2,:],
           'Disc_loss':losses_avg[3,:], 'Reff_disc_loss': losses_avg[4,:]}
    savemat('{}/losses-{}.mat'.format(model_name, datetime.datetime.now().strftime('%m.%d--%H:%M:%S')),dic)
    savemat('{}/lr_rates-{}.mat'.format(model_name, datetime.datetime.now().strftime('%m.%d--%H:%M:%S')),lr_rates)
    if not discriminator_reff: # if not discriminator_reff:
        discriminator_reff = discriminator
        generator = Generator()
        discriminator = Discriminator()
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate_gen, beta_1=BETA_1_GEN, beta_2=BETA_2_GEN)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_disc, beta_1=BETA_1_DISC, beta_2=BETA_2_DISC)
        losses_val = np.zeros((4,0))
        losses_avg = np.zeros((5,0)) 
        learning_rates = np.zeros((2,0))
        fit(train_sequence, epochs= 150, model_name=model_name)
        # Y_TRAIN_SIZE = 2
        # fit(train_sequence, epochs= 50, model_name=model_name)    
        dic = {'Gen_total_loss': losses_avg[0,:], 'Gen_loss': losses_avg[1,:], 'Gen_l1_loss':losses_avg[2,:],
               'Disc_loss':losses_avg[3,:], 'Reff_disc_loss': losses_avg[4,:]
               }
        lr_rates =  {'gen_lr': learning_rates[0,:], 'disc_lr': learning_rates[1,:]}
        generator.save(model_name+'/generator')
        discriminator.save(model_name+'/discriminator' )
        savemat('{}/losses-{}.mat'.format(model_name, datetime.datetime.now().strftime('%m.%d--%H:%M:%S')),dic)
        savemat('{}/lr_rates-{}.mat'.format(model_name, datetime.datetime.now().strftime('%m.%d--%H:%M:%S')),lr_rates)

    
# [Gen_total_loss, Gen_loss, Gen_l1_loss, Disc_loss, Reff_disc_loss]

# prediction = predict_future(x_train[505])
# generate_image(tf.concat([x_train[505],prediction],axis=-1), y_train[509])
# plt.figure()
# generate_image(x_train[-1],y_train[-1], generator)



