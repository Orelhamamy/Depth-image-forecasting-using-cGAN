#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

def loaddata():
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    
    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',origin=(_URL), extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip),'facades/')
    return PATH

def load(img_file):
    image = tf.io.read_file(img_file)
    image = tf.image.decode_jpeg(image)
    
    width = tf.shape(image)[1]
    width = width // 2
    
    real_img = image[:,:width,:]
    input_img = image[:,width:,:]
    
    real_img = tf.cast(real_img, tf.float32)
    input_img = tf.cast(input_img, tf.float32)
    
    return input_img, real_img

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

PATH  = loaddata()
## ---------- Display image from data set ----------
input_img, real_img = load(PATH + 'train/100.jpg')
# plt.figure()
# plt.imshow(real/255.0)
# plt.figure()
# plt.imshow(in_img/255.0)

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_img = tf.stack([input_image, real_image], axis=0)
    stacked_img = tf.image.random_crop(stacked_img, size = [2 , IMG_HEIGHT, IMG_WIDTH, 3])
    return stacked_img[0], stacked_img[1]

def normalize(input_image, real_image):
    input_image = input_image/127.5 - 1
    real_image = real_image/127.5 - 1
    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    
    input_image, real_image = random_crop(input_image, real_image)
    
    if tf.random.uniform(())>0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    
    return input_image, real_image

## -------------Print 4 example----------------
# plt.figure(figsize = (6,6))
# for i in range(4):
#     inp_jit, re_jit = random_jitter(input_img, real_img)
#     plt.subplot(2,2, i+1)
#     plt.imshow(inp_jit/255.0)
#     plt.axis('off')
# plt.show()

def load_image_train(img_file):
    input_img, output_image = load(img_file)
    input_img, output_image = random_jitter(input_img, output_image)
    input_img, output_image = normalize(input_img, output_image)
    return input_img, output_image

def load_image_test(img_file):
    input_img, output_image = load(img_file)
    input_img, output_image = resize(input_img, output_image, IMG_HEIGHT, IMG_WIDTH)
    input_img, output_image = normalize(input_img, output_image)
    return input_img, output_image

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHENNELS = 3

def downsample(filters, size, apply_batchmorm = True):
    initializer = tf.random_normal_initializer(0.,0.02)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchmorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU(0.2))
    
    return result    

def upsample(filters, size, apply_dropout = False):
    initializer = tf.random_normal_initializer(0.,0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides = 2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(.5))
        
    result.add(tf.keras.layers.ReLU())
    
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape =[IMG_HEIGHT, IMG_WIDTH , 3])
    
    down_stack = [
        downsample(64,4,apply_batchmorm=False), # (bs,128,128,64)
        downsample(128,4), # (bs,64,64,128)
        downsample(256,4), # (bs, 32,32,256)
        downsample(512,4), # (bs, 16,16,512)
        downsample(512,4), # (bs, 8,8,512)
        downsample(512,4), # (bs, 4,4,512)
        downsample(512,4), # (bs, 2,2,512)
        downsample(512,4)] # (bs, 1,1,512)
    
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2,2,512)
        upsample(512, 4, apply_dropout=True), # (bs, 4,4,512)
        upsample(512, 4, apply_dropout=True), # (bs, 8,8,512)
        upsample(512, 4), # (bs, 16,16,512)
        upsample(256,4), # (bs, 32,32,256)
        upsample(128,4), # (bs, 64,64,128)
        upsample(64,4)] # (bs, 128,128,64)
    
    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHENNELS, 4, strides=2, 
                                           padding='same',
                                           activation ='tanh', 
                                           kernel_initializer= initializer)
    
    x = inputs
    connections = []
    for down in down_stack:
        x = down (x)
        connections.append(x)
    connections = reversed(connections[:-1])
    
    for up, conc in zip(up_stack, connections):
        x = up (x)
        x = tf.keras.layers.Concatenate()([conc, x])
    x = last (x)
    
    return tf.keras.Model(inputs = inputs, outputs = x)

generetor = Generator()
tf.keras.utils.plot_model(generetor, show_shapes=True, dpi=64)

gen_output = generetor(input_img[tf.newaxis,...], training= False)
plt.imshow(gen_output[0,...])

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generted_output, gen_output, target):
    gen_loss = loss_object(tf.ones_like(disc_generted_output), disc_generted_output)
    
    l1_loss = tf.reduce_mean(tf.abs(gen_output-target))
    total_gen_loss = gen_loss + LAMBDA*l1_loss
    
    return total_gen_loss, gen_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0.,0.02)
    
    inp = tf.keras.layers.Input(shape= [IMG_HEIGHT,IMG_WIDTH,3], name = 'input_img') # (bs, 256,256,3)
    tar = tf.keras.layers.Input(shape= [IMG_HEIGHT,IMG_WIDTH,3], name = 'target_img') # (bs, 256,256,3)
    conc = tf.keras.layers.concatenate([inp, tar], axis = -1) # (bs, 256,256,6)
    
    down1 = downsample(64,4, False)(conc) # (bs, 128,128,64)
    down2 = downsample(128,4)(down1) # (bs, 64,64,128)
    down3 = downsample(256,4)(down2) # (bs, 32,32,256)
    
    zero_ped1 = tf.keras.layers.ZeroPadding2D(padding=1)(down3) # (bs, 34,34,256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, 
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_ped1) # (bs, 31,31,512)
    
    batch_norm = tf.keras.layers.BatchNormalization() (conv) # (bs, 31,31,512)
    active_relu = tf.keras.layers.LeakyReLU(0.2)(batch_norm) # (bs, 31,31,512)
    zero_ped2 = tf.keras.layers.ZeroPadding2D(padding=1)(active_relu) # (bs, 33,33,512)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer = initializer,
                                  use_bias = False) (zero_ped2)
    
    return tf.keras.Model(inputs = [inp, tar],outputs = last)

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, to_file='Discriminator.png',show_shapes=True, dpi=64)

disc_out = discriminator([input_img[tf.newaxis,...],gen_output], training = False)
plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap = 'RdBu_r')
plt.colorbar()


def discriminator_loss(disc_real_output, disc_gen_output):
    real_loss = loss_object(tf.ones_like(disc_real_output),disc_real_output)
    
    gen_loss = loss_object(tf.zeros_like(disc_real_output), disc_gen_output)
    
    return real_loss + gen_loss

generetor_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=.999)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=.999)

checkpoint_dir = './traning_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(generetor_optimizer=generetor_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generetor = generetor,
                                 discriminator = discriminator)


def generate_images(model, input_img, target):
    prediction = model(input_img, training = True)
    
    disp = [input_img[0], target[0], prediction[0]]
    titles = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    plt.figure(figsize = (15,15))
    for i in range(3): 
        plt.subplot(1,3,i+1)
        plt.imshow(disp[i]*0.5+0.5)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

## ---------------- generate example images ----------- 
# for example_input, example_tar in test_dataset.take(1):
#     generate_images(generetor, example_input, example_tar)

import datetime

log_dir = 'logs/'

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/"+ datetime.datetime.now().strftime('%Y.%m.%d--%H:%M:%S'))

@tf.function()
def train_step(input_img, target, epoch):
    with tf.GradientTape() as  gen_tape , tf.GradientTape() as disc_tape:
        gen_output = generetor(input_img, training= True)
        disc_gen_output = discriminator([input_img, gen_output], training = True)
        disc_real_output = discriminator([input_img, target], training = True)
        
        gen_total_loss, gen_loss, gen_l1_loss = generator_loss(disc_gen_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_gen_output)      
        
        gen_gradient = gen_tape.gradient(gen_total_loss, generetor.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        generetor_optimizer.apply_gradients(zip(gen_gradient, 
                                                generetor.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradient,
                                                    discriminator.trainable_variables))
        
    with summary_writer.as_default():
        tf.summary.scalar('Gen_total_loss', gen_total_loss, step = epoch)
        tf.summary.scalar('Gen_loss', gen_loss, step=epoch)
        tf.summary.scalar('Gen_l1_loss', gen_l1_loss, step= epoch)
        tf.summary.scalar('Disc_loss', disc_loss, step = epoch)

EPOCHS = 150

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        print('Epoch:', epoch+1)
        # for example_input, example_tar in test_ds.take(1):
        #     generate_images(generetor, example_input, example_tar)
        
        for n, (input_img, target_img) in train_ds.enumerate():
            train_step(input_img, target_img, epoch)
            print('.', end='')
            if (n+1)%100==0:
                print()
        print()
        if (epoch+1)%20==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken to epoch: {} in sec {}\n'.format(epoch+1, time.time()-start))
    
    checkpoint.save(file_prefix=checkpoint_prefix)
discriminator.summary()   
generetor.summary()
# fit(train_dataset, EPOCHS, test_dataset)












