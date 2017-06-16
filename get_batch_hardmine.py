#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:18:27 2017

@author: cvpr
"""
import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def cnn_get_batch_hardmine(imagePaths, imageSizes, labelRects):
    images = []
    clsmaps = []
    regmaps = []
    
    imageCells = [mpimg.imread(i) for i in imagePaths]
    #print imageCells[0].shape, imageCells[1].shape
    pasteBox = np.zeros(len(imagePaths), 4)
    for i in range(12):
        imageSize = imageSizes[i]
        labelRect = labelRects[i]
        
        rnd = np.random.rand(1)
        if rnd < 0.33:
            imageSize = math.floor(imageSize / 2)
            labelRect = labelRect / 2
            

with open('imdb.pkl', 'rb') as f :
    imdb = dill.load(f)

imagePath = imdb.name[0 : 12]
imagePath = map(lambda orig_string: imdb.imageDir + orig_string, imagePath)
cnn_get_batch_hardmine(imagePath, imdb.size[0 : 12], imdb.rects[0 : 12])