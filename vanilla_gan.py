# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:11:15 2019

@author: tanma
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,LeakyReLU,Reshape,Conv2D,Conv2DTranspose
from keras.layers import Flatten,Dropout
from keras.optimizers import RMSprop
from keras.models import Model
from keras.preprocessing import image

latent_dim = 100
height = 28
width = 28
channels = 1

with open("image_arr.pkl","rb") as file:
    input_train = pickle.load(file)

generator_input = Input(shape=(latent_dim,))
x = Dense(128 * 28 * 28)(generator_input)
x = LeakyReLU()(x)
x = Reshape((28, 28, 128))(x)
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = Model(generator_input, x)
generator.summary()

discriminator_input = Input(shape=(height, width, channels))
x = Conv2D(128, 3)(discriminator_input)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=1)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=1)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=1)(x)
x = LeakyReLU()(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan_optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
gan.summary()

x_train = input_train

x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
        
iterations = 10000
batch_size = 20

save_dir = 'results/'

start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0

    if step % 100 == 0:
        gan.save_weights('gan.h5')

        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_graph' + str(step) + '.png'))

        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_graph' + str(step) + '.png'))

        
random_latent_vectors = np.random.normal(size=(10, latent_dim))

generated_images = generator.predict(random_latent_vectors)

for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
    
plt.show()