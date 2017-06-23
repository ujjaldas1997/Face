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
from compute_dense_overlap import compute_dense_overlap

def cnn_get_batch_hardmine(imagePaths, imageSizes, labelRects):
    #imageSizes = imdb.size[0 : 12]
    #labelRects = imdb.rects[0 : 12]
    with open('Ref_box_25.pkl', 'rb') as f :
        centers = dill.load(f)
    negThres = 0.3
    posThres = 0.7
    images = np.zeros((500, 500, 3, 12))
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
        paste_y1 = np.random.randint(500 - crop_h + 1)
        paste_x1 = np.random.randint(500 - crop_w + 1)
        paste_y2 = paste_y1 + crop_h
        paste_x2 = paste_x1 + crop_w
        pasteBoxes[i, :] = [paste_x1, paste_y1, paste_x2, paste_y2]
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
            labelRect[:, 0 : 2] += [paste_x1, paste_y1]
        
        '''fig, ax = plt.subplots(1)
        ax.imshow(img)
        for j in range(labelRect.shape[0]):
            rect = patches.Rectangle((labelRect[j, 0], labelRect[j, 1]), labelRect[j, 2], labelRect[j, 3], linewidth = 1, edgecolor = 'b', facecolor = 'none')
            ax.add_patch(rect)
        plt.show()'''
        imageSizes[i] = np.array([500, 500])
        imageCells[i] = img
        labelRects[i] = labelRect
        
    offset_x = -1
    offset_y = -1
    stride_x = 8
    stride_y = 8
    
    varsize_x = 63
    varsize_y = 63
    
    coarse_xx = np.tile(np.array(range(-1, 63 * 8 - 1, 8)), (63, 1))
    coarse_yy = np.tile(np.array([range(-1, 63 * 8 - 1, 8)]).transpose(), (1, 63))
    dx1 = centers[:, 0].reshape(1, 1, 25)
    dy1 = centers[:, 1].reshape(1, 1, 25)
    dx2 = centers[:, 1].reshape(1, 1, 25)
    dy2 = centers[:, 1].reshape(1, 1, 25)
    
    coarse_xx1 = coarse_xx[..., None] + dx1
    coarse_yy1 = coarse_yy[..., None] + dy1
    coarse_xx2 = coarse_xx[..., None] + dx2
    coarse_yy2 = coarse_yy[..., None] + dy2
    
    pad_viomasks = np.zeros((63, 63, 25, 12))
    for i in range(12) :
        padx1 = coarse_xx1 < pasteBoxes[i, 0]
        pady1 = coarse_yy1 < pasteBoxes[i, 1]
        padx2 = coarse_xx2 > pasteBoxes[i, 2]
        pady2 = coarse_yy2 > pasteBoxes[i, 3]
        pad_viomasks[:, :, :, i] = padx1 | pady1 | padx2 | pady2
    
    if len(labelRects) > 0 :
        clsmaps = -np.ones((63, 63, 25, 12))
        regmaps = np.zeros((63, 63, 100, 12))
    
    for i in range(12) :
        imt = imageCells[i]
        labelRect = labelRects[i]
        images[:, :, :, i] = imt
        iou = np.array([])
        
        ng = labelRect.shape[0] 
        if ng > 0 :
            gx1 = labelRect[:, 0]   #Use carefully, gx2 is width and gy2 is height
            gy1 = labelRect[:, 1]
            gx2 = labelRect[:, 2]
            gy2 = labelRect[:, 3]
            iou = compute_dense_overlap(offset_x, offset_y, stride_x, stride_y, varsize_x, varsize_y, dx1, dy1, dx2, dy2, gx1, gy1, gx2, gy2)
            fxx1 =  labelRect[:, 0].reshape(1, 1, 1, ng)
            fyy1 =  labelRect[:, 1].reshape(1, 1, 1, ng)
            fxx2 =  (labelRect[:, 2] + labelRect[:, 0]).reshape(1, 1, 1, ng)
            fyy2 =  (labelRect[:, 3] + labelRect[:, 1]).reshape(1, 1, 1, ng)
            
            dhh = dy2 - dy1 + 1
            dww = dx2 - dx1 + 1
            fcx = (fxx1 + fxx2) / 2
            fcy = (fyy1 + fyy2) / 2
            tx = (fcx - coarse_xx) / dww
            ty = (fcy - coarse_yy) / dhh
            fhh = fyy2 - fyy1 + 1
            fww = fxx2 - fxx1 + 1
            tw = np.log(fww / dww)
            th = np.log(fhh / dhh)
        if len(iou) > 0:
            iou = iou + 1e6 * np.random.rand(iou.shape)
        clsmap = -np.ones((63, 63, 25))
        regmap = np.zeros((63, 63, 100))
        if ng > 0 :
            best_iou = np.amax(iou, axis = 3)
            best_face_per_loc = np.argmax(iou, axis = 3)
            regidx = np.ravel_multi_index(np.array((np.array(range(63*63*25)), np.ravel(best_face_per_loc, order = 'F'))), dims = (63 * 63 * 25), order = 'F')
            tx = np.reshape(tx.ravel(order = 'F')[regidx], 63, 63, 25)
            ty = np.reshape(ty.ravel(order = 'F')[regidx], 63, 63, 25)
            temp = np.ones((63, 63, 1, ng))
            for j in range(ng) :
                temp[:, :, 1, j] *= tw[0, 0, 0, j]
            tw[:] = temp
            tw = np.reshape(tw.ravel(order = 'F')[regidx], 63, 63, 25)
            temp = temp = np.ones((63, 63, 1, ng))
            for j in range(ng) :
                temp[:, :, 1, j] *= th[0, 0, 0, j]
            th[:] = temp
            th = np.reshape(th.ravel(order = 'F')[regidx], 63, 63, 25)
            regmap = np.concatenate((tx, ty, tw, th), axis = 2)
            
            temp = iou.reshape(8, 3)[[0, 4, 2, 6, 1, 5, 3, 7], :]
            iou_ = np.amax(temp, axis = 0)
            fbest_idx = np.argmax(temp, axis = 0)
            clsmap.ravel(order = 'F')[fbest_idx(iou_ > negThres)] = 1
            clsmap = np.maximum(clsmap, (best_iou >= posThres) * 2 - 1)
            gray = -np.ones(clsmap.shape)
            gray = gray * np.logical_and(best_iou >= negThres, best_iou < posThres)
            clsmap = np.maximum(clsmap, gray)
        clsmap = clsmap * np.logical_and(pad_viomasks[:, :, :, i], clsmap != -1)
        regmap[:, :, 0 : 25] = regmap[:, :, 0 : 25] * np.logical_and(pad_viomasks[:, :, :, i], clsmap != -1)
        
        clsmaps[:, :, :, i] = clsmap
        regmaps[:, :, :, i] = regmap
        print clsmap.shape, regmap.shape

with open('imdb.pkl', 'rb') as f :
    imdb = dill.load(f)

imagePaths = imdb.name[0 : 12]
imagePaths = map(lambda orig_string: imdb.imageDir + orig_string, imagePaths)
cnn_get_batch_hardmine(imagePaths, imdb.size[0 : 12], imdb.rects[0 : 12])
