#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:57:38 2017

@author: ujjaldas223
"""
import numpy as np
def compute_dense_overlap(offset_x, offset_y, stride_x, stride_y, varsize_x, varsize_y, dx1, dy1, dx2, dy2, gx1, gy1, gx2, gy2) :
    dx1 = dx1[0, 0, :]
    dy1 = dy1[0, 0, :]
    dx2 = dx2[0, 0, :]
    dy2 = dy2[0, 0, :]
    ng = len(gx1)
    nt = len(dx1)
    iou = np.zeros((varsize_y, varsize_x, nt, ng))
    tmp_overlap = np.zeros((varsize_y, varsize_x, nt))
    for i in range(ng) :
        bbox_x1 = gx1[i]
        bbox_y1 = gy1[i]
        bbox_w = gx2[i]
        bbox_h = gy2[i]
        bbox_x2 = bbox_x1 + bbox_w
        bbox_y2 = bbox_y1 + bbox_h
        bbox_area = bbox_h * bbox_w
        for j in range(nt) :
            delta_x1 = dx1[j]
            delta_y1 = dy1[j]
            delta_x2 = dx2[j]
            delta_y2 = dy2[j]
            filter_h = delta_y2 - delta_y1 + 1
            filter_w = delta_x2 - delta_x1 + 1
            filter_area = filter_h * filter_w
            tidx = 63 * 63 * j
            
            for x in range(63) :
                for y in range(63) :
                        cx = offset_x + x * stride_x
                        xidx = 63 * x
                        cy = offset_y + y * stride_y
                        
                        x1 = delta_x1 + cx
                        y1 = delta_y1 + cy
                        x2 = delta_x2 + cx
                        y2 = delta_y2 + cy
                        xx1 = max(x1, bbox_x1)
                        yy1 = max(y1, bbox_y1)
                        xx2 = min(x2, bbox_x2)
                        yy2 = min(y2, bbox_y2)
                        int_w = xx2 - xx1 + 1
                        int_h = yy2 - yy1 + 1
                        
                        if int_w > 0 and int_h > 0 :
                            int_area = int_w * int_h
                            union_area = filter_area + bbox_area - int_area
                            tmp_overlap[tidx + xidx + y] = int_area / union_area
                        else :
                            tmp_overlap[tidx + xidx + y] = 0
        iou[:, :, :, i] = tmp_overlap
    return iou