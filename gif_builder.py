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