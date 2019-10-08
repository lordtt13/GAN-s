# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 00:17:49 2019

@author: tanma
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model

import numpy as np
import argparse

import gan


def train(models, data, params):
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_label = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_label, axis=1))

    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_images = generator.predict([noise, fake_labels])
        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((real_labels, fake_labels))

        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        metrics  = discriminator.train_on_batch(x, [y, labels])
        fmt = "%d: [disc loss: %f, srcloss: %f, lblloss: %f, srcacc: %f, lblacc: %f]" 
        log = fmt % (i, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
        
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        y = np.ones([batch_size, 1])

        metrics  = adversarial.train_on_batch([noise, fake_labels], [y, fake_labels])
        fmt = "%s [advr loss: %f, srcloss: %f, lblloss: %f, srcacc: %f, lblacc: %f]" 
        log = fmt % (log, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            
            gan.plot_images(generator,
                        noise_input=noise_input,
                        noise_label=noise_label,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)

    generator.save(model_name + ".h5")


def build_and_train_models():
    (x_train, y_train), (_, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)

    model_name = "acgan_mnist"
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )

    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = gan.discriminator(inputs, num_labels=num_labels)
    optimizer = RMSprop(lr=lr, decay=decay)
    loss = ['binary_crossentropy', 'categorical_crossentropy']
    discriminator.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    labels = Input(shape=label_shape, name='labels')
    generator = gan.generator(inputs, image_size, labels=labels)
    generator.summary()

    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False
    adversarial = Model([inputs, labels],
                        discriminator(generator([inputs, labels])),
                        name=model_name)
    
    adversarial.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 10))
        noise_label[:,class_label] = 1
        step = class_label

    gan.plot_images(generator,
                    noise_input=noise_input,
                    noise_label=noise_label,
                    show=True,
                    step=step,
                    model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        if args.digit is not None:
            class_label = args.digit
        test_generator(generator, class_label)
    else:
        build_and_train_models()