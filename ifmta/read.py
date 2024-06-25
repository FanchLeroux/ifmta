# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:42:32 2024

@author: f24lerou
"""
import cv2
import numpy as np

def ReadFramesFromAvi(filename, *, gray=True):
    
    cap = cv2.VideoCapture(str(filename))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = np.zeros((length, height, width), dtype=np.uint8)
    
    for k in range(length):
        ret, frame = cap.read()
        if gray:
            frames[k] = frame[:,:,0]
        else:
            frames[k] = frame
    
    return frames
    