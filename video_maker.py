# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:36:03 2019

@author: tanma
"""

import cv2
import os

image_folder = 'C://Users//tanma.TANMAY-STATION//Desktop//GitHub//GAN//cgan_mnist//'
video_name = 'video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()