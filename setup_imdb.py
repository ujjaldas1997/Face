#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:28:19 2017

@author: cvpr
"""

from scipy.io import loadmat
import magic
import re
import dill

train_data = loadmat('wider_face_split/wider_face_train.mat', matlab_compatible = False, struct_as_record = False)
class my_imdb :
    def __init__(self, imageDir) :
        self.imageDir = imageDir
        self.name = []
        self.size = []
        self.imageSet = []
        self.rects = []
        self.eventId = []

imdb = my_imdb('data/WIDER_train/images/')
count = 0
for i in range(train_data['event_list'].size):
    imageDir = imdb.imageDir + str(train_data['event_list'][i][0])[3:-2] + '/'
    imageList = train_data['file_list'][i][0]
    bboxList = train_data['face_bbx_list'][i][0]
    for j in range(imageList.size):
        count += 1
        imagePath = str(imageList[j][0][0]) + '.jpg'
        imdb.name.append(str(train_data['event_list'][i][0])[3:-2] + '/' + imagePath)
        
        height, width = re.findall('(\d+)x(\d+)', magic.from_file(imageDir + imagePath))[1]
        imdb.size.append([int(height), int(width)])
        
        imdb.imageSet = 1
        imdb.rects.append(bboxList[j][0])
        imdb.eventId.append(i)
        
with open('imdb.pkl', 'wb') as f :
    dill.dump(imdb, f)
