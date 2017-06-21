#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:33:37 2017

@author: ujjaldas223
"""
import dill
import numpy as np
import pam

with open('imdb.pkl', 'rb') as f :
    imdb = dill.load(f)
N = 25
rects = np.vstack(imdb.rects)

hs = rects[:, 3]
ws = rects[:, 2]
rects = np.vstack((hs, ws)).T
rects = np.delete(rects, np.where(rects[:, 0] < 10)[0], axis = 0)
rects = np.delete(rects, np.where(rects[:, 1] < 10)[0], axis = 0)
hs = rects[:, 0]
ws = rects[:, 1]
rects = np.vstack((-(ws - 1) / 2, -(hs - 1) / 2, (ws - 1) / 2, (hs - 1) / 2)).T

np.random.shuffle(rects)
rects = rects[0 : 100000, :]

best_cost, best_choice, best_medoids = pam.kmedoids(rects, 25)
rects = rects[best_choice]

index = np.argsort(rects[:, 2] * rects[:, 3])[: : -1]
rects = rects[index, :]

with open('Ref_box_25.pkl', 'wb') as f :
    dill.dump(rects, f)