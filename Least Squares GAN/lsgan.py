# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:28:43 2019

@author: tanma
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import argparse

import gan


def build_and_train_models():
    (x_train, _), (_, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    model_name = "lsgan_mnist"
    latent_size = 100
    input_shape = (image_size, image_size, 1)
    batch_size = 64
    lr = 2e-4
    decay = 6e-8
    train_steps = 20000

    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = gan.discriminator(inputs, activation=None)
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = gan.generator(inputs, image_size)
    generator.summary()

    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    adversarial = Model(inputs,
                        discriminator(generator(inputs)),
                        name=model_name)
    adversarial.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    gan.train(models, x_train, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        gan.test_generator(generator)
    else:
        build_and_train_models()