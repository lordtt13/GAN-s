# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:30:42 2019

@author: tanma
"""

from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from numba import jit

import numpy as np
import argparse
import gan


@jit()
def train(models, data, params):
    generator, discriminator, adversarial = models

    x_train, y_train = data

    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 100
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_label = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_label, axis=1))

    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]

        real_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        real_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])
        inputs = [noise, fake_labels, fake_code1, fake_code2]
        fake_images = generator.predict(inputs)

        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((real_labels, fake_labels))
        codes1 = np.concatenate((real_code1, fake_code1))
        codes2 = np.concatenate((real_code2, fake_code2))

        y = np.ones([2 * batch_size, 1])

        y[batch_size:, :] = 0

        outputs = [y, labels, codes1, codes2]

        metrics = discriminator.train_on_batch(x, outputs)
        fmt = "%d: [discriminator loss: %f, label_acc: %f]"
        log = fmt % (i, metrics[0], metrics[6])

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        fake_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])

        y = np.ones([batch_size, 1])

        inputs = [noise, fake_labels, fake_code1, fake_code2]
        outputs = [y, fake_labels, fake_code1, fake_code2]
        metrics  = adversarial.train_on_batch(inputs, outputs)
        fmt = "%s [adversarial loss: %f, label_acc: %f]"
        log = fmt % (log, metrics[0], metrics[6])

        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
    
            gan.plot_images(generator,
                            noise_input=noise_input,
                            noise_label=noise_label,
                            noise_codes=[noise_code1, noise_code2],
                            show=show,
                            step=(i + 1),
                            model_name=model_name)
   
    generator.save(model_name + ".h5")


def mi_loss(c, q_of_c_given_x):
    return K.mean(-K.sum(K.log(q_of_c_given_x + K.epsilon()) * c, axis=1))


def build_and_train_models(latent_size=100):
    (x_train, y_train), (_, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)

    model_name = "infogan_mnist"

    batch_size = 64
    train_steps = 5000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )
    code_shape = (1, )

    inputs = Input(shape=input_shape, name='discriminator_input')

    discriminator = gan.discriminator(inputs,
                                      num_labels=num_labels,
                                      num_codes=2)
    optimizer = RMSprop(lr=lr, decay=decay)

    loss = ['binary_crossentropy', 'categorical_crossentropy', mi_loss, mi_loss]

    loss_weights = [1.0, 1.0, 0.5, 0.5]
    discriminator.compile(loss=loss,
                          loss_weights=loss_weights,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    labels = Input(shape=label_shape, name='labels')
    code1 = Input(shape=code_shape, name="code1")
    code2 = Input(shape=code_shape, name="code2")

    generator = gan.generator(inputs,
                              image_size,
                              labels=labels,
                              codes=[code1, code2])
    generator.summary()

    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    discriminator.trainable = False

    inputs = [inputs, labels, code1, code2]
    adversarial = Model(inputs,
                        discriminator(generator(inputs)),
                        name=model_name)

    adversarial.compile(loss=loss,
                        loss_weights=loss_weights,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, params, latent_size=100):
    label, code1, code2, p1, p2 = params
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    step = 0
    if label is None:
        num_labels = 10
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 10))
        noise_label[:,label] = 1
        step = label

    if code1 is None:
        noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p1:
            a = np.linspace(-2, 2, 16)
            a = np.reshape(a, [16, 1])
            noise_code1 = np.ones((16, 1)) * a
        else:
            noise_code1 = np.ones((16, 1)) * code1
        print(noise_code1)

    if code2 is None:
        noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p2:
            a = np.linspace(-2, 2, 16)
            a = np.reshape(a, [16, 1])
            noise_code2 = np.ones((16, 1)) * a
        else:
            noise_code2 = np.ones((16, 1)) * code2
        print(noise_code2)

    gan.plot_images(generator,
                    noise_input=noise_input,
                    noise_label=noise_label,
                    noise_codes=[noise_code1, noise_code2],
                    show=True,
                    step=step,
                    model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Specify latent code 1"
    parser.add_argument("-a", "--code1", type=float, help=help_)
    help_ = "Specify latent code 2"
    parser.add_argument("-b", "--code2", type=float, help=help_)
    help_ = "Plot digits with code1 ranging fr -n1 to +n2"
    parser.add_argument("--p1", action='store_true', help=help_)
    help_ = "Plot digits with code2 ranging fr -n1 to +n2"
    parser.add_argument("--p2", action='store_true', help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        label = args.digit
        code1 = args.code1
        code2 = args.code2
        p1 = args.p1
        p2 = args.p2
        params = (label, code1, code2, p1, p2)
        test_generator(generator, params, latent_size=62)
    else:
        build_and_train_models(latent_size=62)