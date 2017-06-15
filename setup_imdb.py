#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:28:19 2017

@author: cvpr
"""

from scipy.io import loadmat
import magic
import re

train_data = loadmat('wider_face_split/wider_face_train.mat', matlab_compatible = False, struct_as_record = False)
class imdb :
    imageDir = 'data/WIDER_train/images'
    class images :
        name = None
        size = None
        imageSet = 0
    class labels :
        rects = None
        eventId = None

imdb = imdb()
count = 0
for i in range(train_data['event_list'].size):
    imageDir = imdb.imageDir + str(train_data['event_list'][i][0])[3:-2]
    imageList = train_data['file_list'][i][0]
    bboxList = train_data['face_bbx_list'][i][0]
    for j in range(imageList.size):
        count += 1
        imagePath = str(imageList[j][0][0]) + 'jpg'
        imdb.images.name.append(imagePath)
        
        height, width = re.search('(\d+) x (\d+)', magic.from_file(imageDir + imagePath)).groups()
        imdb.images.size.append([height, width])
        
        imdb.images.imageSet = 1
        imdb.images.rects.append(bboxList[i][0])
        imdb.labels.eventId.append(i)