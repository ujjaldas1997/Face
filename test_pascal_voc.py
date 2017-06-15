#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:32:41 2017

@author: cvpr
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
from scipy.misc import bytescale

from keras.models import Model
from keras.layers import Input, Permute, Convolution2D, Deconvolution2D, Cropping2D, merge
from scipy.io import loadmat

data = loadmat('pascal-fcn8s-tvg-dag.mat', matlab_compatible = False, struct_as_record = False)
l = data['layers']
p = data['params']
description = data['meta'][0, 0].classes[0, 0].description
#print data.keys()
#print l.shape, p.shape, description.shape
#class2index = {}
#for i, clsname in enumerate(description[0, :]) :
#   class2index[str(clsname[0])] = i
#print sorted(class2index.keys())
#for i in range(0, p.shape[1]-1-2*2, 2) :
#    print(i,
#          str(p[0,i].name[0]), p[0,i].value.shape,
#          str(p[0,i+1].name[0]), p[0,i+1].value.shape)
#print '------------------------------------------------------'
#for i in range(p.shape[1]-1-2*2+1, p.shape[1]) :
#    print(i,
#          str(p[0,i].name[0]), p[0,i].value.shape)
#for i in range(l.shape[1]):
#    print(i,
#          str(l[0,i].name[0]), str(l[0,i].type[0]),
#          [str(n[0]) for n in l[0,i].inputs[0,:]],
#          [str(n[0]) for n in l[0,i].outputs[0,:]])

