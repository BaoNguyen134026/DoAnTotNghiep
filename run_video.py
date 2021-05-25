
import numpy as np
import cv2
from random import randint
import time
# cap = cv2.
cap1 = cv2.VideoCapture('/home/bao/Desktop/video/source_train/phai.avi')
f = np.load('/home/bao/Desktop/video/source_train/phai.npy')
# f2 = np.load('/home/bao/Desktop/video/sp/trai.npy')

print(cap1.get(7))
# print(f2.shape)
# cap = cv2.VideoCapture()
x,y,w,h = 550,228,30,20
for index in range(int(cap1.get(7))):
    ret,frame = cap1.read()
    frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1)
    cv2.imshow('video',frame)
    
    if cv2.waitKey(30)==27:
        break
