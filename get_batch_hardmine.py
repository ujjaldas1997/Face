#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:18:27 2017

@author: cvpr
"""
import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy.misc as spm

def cnn_get_batch_hardmine(imagePaths, imageSizes, labelRects):
    #imageSizes = imdb.size[0 : 12]
    #labelRects = imdb.rects[0 : 12]
    images = []
    clsmaps = []
    regmaps = []
    
    imageCells = [spm.imread(i) for i in imagePaths]
    pasteBoxes = np.zeros((len(imagePaths), 4))
    for i in range(12):
        imageSize = imageSizes[i]
        labelRect = labelRects[i]
        
        factor = min(imageSize) / 500
        if factor < 1 :
            imageSize = imageSize * 2
            labelRect = labelRect * 2
            img = spm.imresize(imageCells[i], 2, interp = 'bilinear', mode = None)
        elif factor > 2 :
            imageSize = imageSize / 2
            labelRect = labelRect / 2
            img = spm.imresize(imageCells[i], 0.5, interp = 'bilinear', mode = None)
        else:
            img = imageCells[i]
        
        crop_y1 = np.random.randint(max(1, imageSize[1] - 500))
        crop_x1 = np.random.randint(max(1, imageSize[0] - 500))
        crop_y2 = min(imageSize[1], crop_y1 + 500)
        crop_x2 = min(imageSize[0], crop_x1 + 500)
        crop_h = crop_y2 - crop_y1
        crop_w = crop_x2 - crop_x1
        #print "image size = ", imageSize[0], imageSize[1]
        #print crop_h, crop_w, '(', crop_x1, crop_y1, ')', '(', crop_x2, crop_y2, ')'    
        #paste_y1 = np.random.randint(500 - crop_h + 1)
        #paste_x1 = np.random.randint(500 - crop_w + 1)
        #paste_y2 = paste_y1 + crop_h
        #paste_x2 = paste_x1 + crop_w
        #pasteBoxes[i, :] = [paste_x1, paste_y1, paste_x2, paste_y2]
        '''fig, ax = plt.subplots(1)
        ax.imshow(img)
        rect = patches.Rectangle((crop_x1, crop_y1), crop_w, crop_h, linewidth = 1, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)
        for j in range(labelRect.shape[0]):
            rect = patches.Rectangle((labelRect[j, 0], labelRect[j, 1]), labelRect[j, 2], labelRect[j, 3], linewidth = 1, edgecolor = 'b', facecolor = 'none')
            ax.add_patch(rect)
        plt.show()'''
        img = img[crop_y1 : crop_y2, crop_x1 : crop_x2, :]
        if len(labelRect) > 0:
            
            labelRect = np.delete(labelRect, np.where(labelRect[:, 0] < crop_x1)[0], axis = 0)
            labelRect = np.delete(labelRect, np.where(labelRect[:, 1] < crop_y1)[0], axis = 0)
            labelRect = np.delete(labelRect, np.where((labelRect[:, 0 : 2] + labelRect[:, 2 : ])[:, 0] >= crop_x2)[0], axis = 0)
            labelRect = np.delete(labelRect, np.where((labelRect[:, 0 : 2] + labelRect[:, 2 : ])[:, 1] >= crop_y2)[0], axis = 0)
            
            labelRect[:, 0 : 2] -= [crop_x1, crop_y1]
            #labelRect[:, 0 : 2] += [paste_x1, paste_y1]
        
        '''fig, ax = plt.subplots(1)
        ax.imshow(img)
        for j in range(labelRect.shape[0]):
            rect = patches.Rectangle((labelRect[j, 0], labelRect[j, 1]), labelRect[j, 2], labelRect[j, 3], linewidth = 1, edgecolor = 'b', facecolor = 'none')
            ax.add_patch(rect)
        plt.show()'''
        imageCells[i] = img
        labelRects[i] = labelRect
        
    

with open('imdb.pkl', 'rb') as f :
    imdb = dill.load(f)

imagePaths = imdb.name[0 : 12]
imagePaths = map(lambda orig_string: imdb.imageDir + orig_string, imagePaths)
cnn_get_batch_hardmine(imagePaths, imdb.size[0 : 12], imdb.rects[0 : 12])
