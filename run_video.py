
import numpy as np
import cv2
from random import randint
import time

cap1 = cv2.VideoCapture('/home/bao/Desktop/DoAnTotNghiep/videosource/body_yen_tudo.avi')

for index in range(int(cap1.get(7))):
    ret,frame = cap1.read()
    frame = cv2.rectangle(frame, (322,123), (352,140), (0,0,255), 1)
    cv2.imshow('video',frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
