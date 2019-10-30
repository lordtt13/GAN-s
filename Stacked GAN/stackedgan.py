# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 00:22:37 2019

@author: tanma
"""

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from keras.layers.merge import concatenate
from numba import jit

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import gan

@jit()
def build_encoder(inputs, num_labels=10, feature1_dim=256):
    kernel_size = 3
    filters = 64

    x, feature1 = inputs
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)
    feature1_output = Dense(feature1_dim, activation='relu')(y)

    enc0 = Model(inputs=x, outputs=feature1_output, name="encoder0")
    
    y = Dense(num_labels)(feature1)
    labels = Activation('softmax')(y)

    enc1 = Model(inputs=feature1, outputs=labels, name="encoder1")

    return enc0, enc1


def build_generator(latent_codes, image_size, feature1_dim=256):
    labels, z0, z1, feature1 = latent_codes

    inputs = [labels, z1]
    x = concatenate(inputs, axis=1)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    fake_feature1 = Dense(feature1_dim, activation='relu')(x)

    gen1 = Model(inputs, fake_feature1, name='gen1')

    gen0 = gan.generator(feature1, image_size, codes=z0)

    return gen0, gen1


def build_discriminator(inputs, z_dim=50):
    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)

    f1_source = Dense(1)(x)
    f1_source = Activation('sigmoid', name='feature1_source')(f1_source)

    z1_recon = Dense(z_dim)(x) 
    z1_recon = Activation('tanh', name='z1')(z1_recon)
    
    discriminator_outputs = [f1_source, z1_recon]
    dis1 = Model(inputs, discriminator_outputs, name='dis1')
    return dis1


def train(models, data, params):
    enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1 = models

    batch_size, train_steps, num_labels, z_dim, model_name = params

    (x_train, y_train), (_, _) = data

    save_interval = 100

    z0 = np.random.normal(scale=0.5, size=[16, z_dim])
    z1 = np.random.normal(scale=0.5, size=[16, z_dim])
    noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_params = [noise_class, z0, z1]

    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]

        real_feature1 = enc0.predict(real_images)

        real_z1 = np.random.normal(scale=0.5, size=[batch_size, z_dim])

        real_labels = y_train[rand_indexes]

        fake_z1 = np.random.normal(scale=0.5, size=[batch_size, z_dim])
        fake_feature1 = gen1.predict([real_labels, fake_z1])

        feature1 = np.concatenate((real_feature1, fake_feature1))
        z1 = np.concatenate((fake_z1, fake_z1))

        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        metrics = dis1.train_on_batch(feature1, [y, z1])

        log = "%d: [dis1_loss: %f]" % (i, metrics[0])

        fake_z0 = np.random.normal(scale=0.5, size=[batch_size, z_dim])

        fake_images = gen0.predict([real_feature1, fake_z0])
       
        x = np.concatenate((real_images, fake_images))
        z0 = np.concatenate((fake_z0, fake_z0))
        
        metrics = dis0.train_on_batch(x, [y, z0])

        log = "%s [dis0_loss: %f]" % (log, metrics[0])

        fake_z1 = np.random.normal(scale=0.5, size=[batch_size, z_dim])

        gen1_inputs = [real_labels, fake_z1]

        y = np.ones([batch_size, 1])
    
        metrics = adv1.train_on_batch(gen1_inputs, [y, fake_z1, real_labels])
        fmt = "%s [adv1_loss: %f, enc1_acc: %f]"

        log = fmt % (log, metrics[0], metrics[6])

        fake_z0 = np.random.normal(scale=0.5, size=[batch_size, z_dim])
        gen0_inputs = [real_feature1, fake_z0]

        metrics = adv0.train_on_batch(gen0_inputs, [y, fake_z0, real_feature1])
        log = "%s [adv0_loss: %f]" % (log, metrics[0])

        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            generators = (gen0, gen1)
            plot_images(generators,
                        noise_params=noise_params,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)

    gen1.save(model_name + "-gen1.h5")
    gen0.save(model_name + "-gen0.h5")
    

def plot_images(generators,
                noise_params,
                show=False,
                step=0,
                model_name="gan"):

    gen0, gen1 = generators
    noise_class, z0, z1 = noise_params
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    feature1 = gen1.predict([noise_class, z1])
    images = gen0.predict([feature1, z0])
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_class.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def train_encoder(model, data, model_name="stackedgan_mnist", batch_size=64):

    (x_train, y_train), (x_test, y_test) = data
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=10,
              batch_size=batch_size)

    model.save(model_name + "-encoder.h5")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))


def build_and_train_models():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255 

    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_test = x_test.astype('float32') / 255

    num_labels = len(np.unique(y_train))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model_name = "stackedgan_mnist"
    batch_size = 64
    train_steps = 1000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    z_dim = 50
    z_shape = (z_dim, )
    feature1_dim = 256
    feature1_shape = (feature1_dim, )

    inputs = Input(shape=input_shape, name='discriminator0_input')
    dis0 = gan.discriminator(inputs, num_codes=z_dim)

    optimizer = RMSprop(lr=lr, decay=decay)

    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 10.0] 
    dis0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis0.summary()

    input_shape = (feature1_dim, )
    inputs = Input(shape=input_shape, name='discriminator1_input')
    dis1 = build_discriminator(inputs, z_dim=z_dim )

    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 1.0] 
    dis1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis1.summary() 

    feature1 = Input(shape=feature1_shape, name='feature1_input')
    labels = Input(shape=label_shape, name='labels')
    z1 = Input(shape=z_shape, name="z1_input")
    z0 = Input(shape=z_shape, name="z0_input")
    latent_codes = (labels, z0, z1, feature1)
    gen0, gen1 = build_generator(latent_codes, image_size)
    gen0.summary() 
    gen1.summary() 

    input_shape = (image_size, image_size, 1)
    inputs = Input(shape=input_shape, name='encoder_input')
    enc0, enc1 = build_encoder((inputs, feature1), num_labels)
    enc0.summary() 
    enc1.summary() 
    encoder = Model(inputs, enc1(enc0(inputs)))
    encoder.summary()

    data = (x_train, y_train), (x_test, y_test)
    train_encoder(encoder, data, model_name=model_name)

    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)

    enc0.trainable = False

    dis0.trainable = False
    gen0_inputs = [feature1, z0]
    gen0_outputs = gen0(gen0_inputs)
    adv0_outputs = dis0(gen0_outputs) + [enc0(gen0_outputs)]
    adv0 = Model(gen0_inputs, adv0_outputs, name="adv0")

    loss = ['binary_crossentropy', 'mse', 'mse']
    loss_weights = [1.0, 10.0, 1.0] 
    adv0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv0.summary()

    enc1.trainable = False

    dis1.trainable = False
    gen1_inputs = [labels, z1]
    gen1_outputs = gen1(gen1_inputs)
    adv1_outputs = dis1(gen1_outputs) + [enc1(gen1_outputs)]

    adv1 = Model(gen1_inputs, adv1_outputs, name="adv1")

    loss_weights = [1.0, 1.0, 1.0] 
    loss = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
    adv1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv1.summary()

    models = (enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1)
    params = (batch_size, train_steps, num_labels, z_dim, model_name)
    train(models, data, params)


def test_generator(generators, params, z_dim=50):
    class_label, z0, z1, p0, p1 = params
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    if z0 is None:
        z0 = np.random.normal(scale=0.5, size=[16, z_dim])
    else:
        if p0:
            a = np.linspace(-4.0, 4.0, 16)
            a = np.reshape(a, [16, 1])
            z0 = np.ones((16, z_dim)) * a
        else:
            z0 = np.ones((16, z_dim)) * z0
        print("z0: ", z0[:,0])

    if z1 is None:
        z1 = np.random.normal(scale=0.5, size=[16, z_dim])
    else:
        if p1:
            a = np.linspace(-1.0, 1.0, 16)
            a = np.reshape(a, [16, 1])
            z1 = np.ones((16, z_dim)) * a
        else:
            z1 = np.ones((16, z_dim)) * z1
        print("z1: ", z1[:,0])

    noise_params = [noise_class, z0, z1]

    plot_images(generators,
                noise_params=noise_params,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator 0 h5 model with trained weights"
    parser.add_argument("-g", "--generator0", help=help_)
    help_ = "Load generator 1 h5 model with trained weights"
    parser.add_argument("-k", "--generator1", help=help_)
    # help_ = "Load encoder h5 model with trained weights"
    # parser.add_argument("-e", "--encoder", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Specify z0 noise code (as a 50-dim with z0 constant)"
    parser.add_argument("-z", "--z0", type=float, help=help_)
    help_ = "Specify z1 noise code (as a 50-dim with z1 constant)"
    parser.add_argument("-x", "--z1", type=float, help=help_)
    help_ = "Plot digits with z0 ranging fr -n1 to +n2"
    parser.add_argument("--p0", action='store_true', help=help_)
    help_ = "Plot digits with z1 ranging fr -n1 to +n2"
    parser.add_argument("--p1", action='store_true', help=help_)
    args = parser.parse_args()
    # if args.encoder:
    #    encoder = args.encoder
    #else:
    #    encoder = None
    if args.generator0:
        gen0 = load_model(args.generator0)
        if args.generator1:
            gen1 = load_model(args.generator1)
        else:
            print("Must specify both generators 0 and 1 models")
            exit(0)
        class_label = args.digit
        z0 = args.z0
        z1 = args.z1
        p0 = args.p0
        p1 = args.p1
        params = (class_label, z0, z1, p0, p1)
        test_generator((gen0, gen1), params)
    else:
        build_and_train_models()