#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

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
LR_GAN =2e-4; BETA_1_GEN =0.5; BETA_2_GEN =.999
LR_DISC =2e-4; BETA_1_DISC =0.5; BETA_2_DISC =.999
ITERATION_GEN = 1 ; ITERATION_DISC = 1
log_dir = 'logs/'
checkpoint_dir = './traning_checkpoints'


'''
@tf.autograph.experimental.do_not_convert
def load_image(img_file):
    type(img_file)
    name = str(img_file)
    image = tf.io.read_file(img_file)
    image = tf.image.decode_jpeg(image)
    # cv2.imshow('12', image)
    # num = int(name[name.index('--')+2:name.index('.jpg')])
    
    return image #, num 

train_dataset = train_dataset.map(load_image,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
'''
def read_img(files, size_y_data):
    imgs = []
    for img in range(OBSERVE_SIZE):
        x = tf.keras.preprocessing.image.load_img(data_set_path + files[img][0],
                                                  color_mode='grayscale')
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = x/255.0
        if len(imgs)==0:
            imgs = x
        else:
            imgs = tf.concat([imgs,x], axis=2)
    output = tf.keras.preprocessing.image.load_img(data_set_path + files[OBSERVE_SIZE][0],
                                                    color_mode='grayscale')
    output = tf.keras.preprocessing.image.img_to_array(output)
    for y_img_num in range(1,size_y_data):
        y_output = tf.keras.preprocessing.image.load_img(data_set_path + files[OBSERVE_SIZE+y_img_num][0],
                                                         color_mode='grayscale')
        y_output = tf.keras.preprocessing.image.img_to_array(y_output)
        output = tf.concat([output, y_output], axis = 2)
    output = tf.convert_to_tensor(output) / 255.0
    return imgs, output
    
    
def load_data_with_future_y_train():
    x_train =[]
    y_train =[]
    for i in range(BUFFER_SIZE-OBSERVE_SIZE):
        if file_list[i][1]+OBSERVE_SIZE==file_list[i+OBSERVE_SIZE][1]:
            counter = 1 # Count the maximum size of y_train
            y_train_size = len(file_list)-i-OBSERVE_SIZE if (i+OBSERVE_SIZE+Y_TRAIN_SIZE>=len(file_list)) else Y_TRAIN_SIZE
            while (counter < y_train_size) and (file_list[i+counter+OBSERVE_SIZE-1][1]+1==file_list[i+counter+OBSERVE_SIZE][1]):
                counter+=1
            x, y = read_img(file_list[i:i+OBSERVE_SIZE+counter], counter)
            x_train.append(x)
            y_train.append(y)
    return x_train, y_train

def load_data():
    x_train =[]
    y_train =[]
    for i in range(BUFFER_SIZE-OBSERVE_SIZE):
        if file_list[i][1]+OBSERVE_SIZE==file_list[i+OBSERVE_SIZE][1]:
            x, y = read_img(file_list[i:i+OBSERVE_SIZE+1],1)
            x_train.append(x)
            y_train.append(y)
    return x_train, y_train

x_train, y_train = load_data()        

def generate_image(input_imgs, output_img, model = False, training = False):
    if model:
        prediction = model(input_imgs[tf.newaxis, ...], training= training)
        plt.subplot(3,5,14)
        plt.imshow(prediction[0,...], cmap = 'gray')
        plt.axis('off')
        plt.title('Predict')
    for i in range(tf.shape(input_imgs)[2]):
        axs = plt.subplot(3,5,i+1)
        axs.imshow(input_imgs[:,:,i], cmap = 'gray')
        plt.title(i+1)
        plt.axis('off')
    plt.subplot(3,5,13)
    plt.imshow(output_img, cmap = 'gray')
    plt.title('Output')
    plt.axis('off')
    plt.show()
    
# ----------------- Random generate imgs -------------------------    
inx = tf.random.uniform([1],0,len(x_train),dtype = tf.dtypes.int32).numpy()[0]
# generate_image(x_train[inx],y_train[inx])

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
# generator = Generator()
generator = tf.keras.models.load_model('cGAN_5Pic_1y_train/generator')
tf.keras.utils.plot_model(generator, show_shapes=True, 
                          dpi = 96, to_file='Generator.png')


def generator_loss(disc_generated_output, generated_output, target):
    gen_loss = loss_object(tf.ones_like(disc_generated_output),disc_generated_output)
    
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
    
    
# discriminator = Discriminator()
discriminator = tf.keras.models.load_model('cGAN_5Pic_1y_train/discriminator')
# discriminator = tf.keras.models.load_model('cGAN_10Pic_1y_train/discriminator/')
tf.keras.utils.plot_model(discriminator, show_shapes = True,
                            dpi = 96, to_file = 'Discriminator.png')    

def discriminator_loss(disc_generated_output, disc_real_outpput):
        
    real_loss = loss_object(tf.ones_like(disc_real_outpput), disc_real_outpput)
    gen_loss = loss_object(tf.zeros_like(disc_real_outpput), disc_generated_output)
    
    return real_loss + gen_loss
    
    real_loss = loss_object(tf.ones_like(disc_real_outpput), disc_real_outpput)
    gen_loss = loss_object(tf.zeros_like(disc_real_outpput), disc_generated_output)
    
    return real_loss + gen_loss

generator_optimizer = tf.keras.optimizers.Adam(LR_GAN, beta_1=BETA_1_GEN, beta_2=BETA_2_GEN)
discriminator_optimizer = tf.keras.optimizers.Adam(LR_DISC, beta_1=BETA_1_DISC, beta_2=BETA_2_DISC)
# gen_img = generator(x_train[0][tf.newaxis, ...], training=False)
# disc_out = discriminator([x_train[0][tf.newaxis, ...], tf.expand_dims(y_train[0],axis=0)] ,training = False)

# ----------- Print exmaple of input and predict -------------- 
inx = tf.random.uniform([1],0,len(x_train),dtype = tf.dtypes.int32).numpy()[0]
# plt.figure()
# generate_image(x_train[inx], y_train[inx], model = generator)
# plt.figure()
# plt.imshow(disc_out[0,...], cmap = 'RdBu_r')

# ------------------------------------------------------------

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/"+ datetime.datetime.now().strftime('%Y.%m.%d--%H:%M:%S'))

# checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt') # checkpoint_dir = './traning_checkpoints'
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer = discriminator_optimizer,
#                                  generator = generator,
#                                  discriminator = discriminator)


@tf.function
def train_step(input_imgs, target, epoch, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # gen_output = generator(input_imgs, training = True)
        # disc_gen_output = discriminator([input_imgs, gen_output], training = True) 
        
                
        # gradient_disc = disc_tape.gradient(disc_loss,
        #                                    discriminator.trainable_varibles)
        for train in range(ITERATION_GEN):
            gen_output = generator(input_imgs, training = True)
            disc_gen_output = discriminator([input_imgs, gen_output], training = True)
            gen_tot_loss, gen_loss, gen_l1_loss = generator_loss(disc_gen_output,
                                                             gen_output,
                                                             target)
            gradient_gen = gen_tape.gradient(gen_tot_loss, 
                                         generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradient_gen,
                                                    generator.trainable_variables))
        
        for train_disc in range(ITERATION_DISC):
            disc_real_output = discriminator([input_imgs, target], training = True)
            disc_loss = discriminator_loss(disc_gen_output, disc_real_output)
            
            gradient_disc = disc_tape.gradient(disc_loss,
                                           discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradient_disc,
                                                        discriminator.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('Gen_total_loss', gen_tot_loss, step = epoch+ step)
        tf.summary.scalar('Gen_loss', gen_loss, step = epoch + step)
        tf.summary.scalar('Gen_l1_loss', gen_l1_loss, step = epoch + step)
        tf.summary.scalar('Disc_loss', disc_loss, step = epoch + step)

def fit(x_train, y_train, test_ds = False, epochs = EPOCHS, step = 0):
    for epoch in range(epochs):
        start = time.time()
        print('Epoch:', epoch+1)
        if test_ds:
            True # Need to add test data
        for n , (input_imgs, target_img) in enumerate(zip(x_train, y_train)):
            # train_step(input_imgs[tf.newaxis,...], target_img[tf.newaxis,...], epoch)
                for i in range(tf.shape(y_train)[3]):
                    train_step(input_imgs[tf.newaxis,...], target_img[tf.newaxis,:,:,i], epoch, step)
                    gen_img = generator(input_imgs[tf.newaxis,...])
                    input_imgs = tf.concat([input_imgs, gen_img[0]], axis=-1)
                    input_imgs = input_imgs[:,:,:10]
                print('.' ,end='')
                if (n+1)%100==0:
                    print('\n')
        if (epoch+1)%20==0:
            generator.save('cGAN_5Pic_1y_train/generator')
            discriminator.save('cGAN_5Pic_1y_train/discriminator')
        print('\nTime taken to epoch: {} in sec {}\n'.format(epoch+1, time.time()- start))
    generator.save('cGAN_5Pic_1y_train/generator')
    discriminator.save('cGAN_5Pic_1y_train/discriminator')


c=0
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/"+ datetime.datetime.now().strftime('%Y.%m.%d--%H:%M:%S'))
for _ in range(10):
    fit(x_train[:-5], y_train[:-5], epochs=100, step =c*200) 
    c+=1


def predict_future(input_imgs):
    predict_imgs = input_imgs
    for predict in range(5): 
        gen_img = generator(predict_imgs[tf.newaxis,...])
        predict_imgs = tf.concat([predict_imgs, gen_img[0]], axis=-1)
        predict_imgs = predict_imgs[:,:,-OBSERVE_SIZE:]
    
    return predict_imgs                                         

# prediction = predict_future(x_train[505])
# generate_image(tf.concat([x_train[505],prediction],axis=-1), y_train[509])
# plt.figure()
# generate_image(x_train[-1],y_train[-1], generator)



