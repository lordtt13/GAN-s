# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:44:19 2019

@author: tanma
"""

import imageio
from os import listdir
from os.path import isfile, join

mypath = "C://Users//tanma.TANMAY-STATION//Desktop//GitHub//GAN//cgan_mnist//"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

images = []
for filename in files:
    images.append(imageio.imread(mypath+filename))
imageio.mimsave('cgan_mnist.gif', images)

with imageio.get_writer('cgan_mnist_alt.gif', mode='I') as writer:
    for filename in files:
        image = imageio.imread(mypath+filename)
        writer.append_data(image)