#!/usr/bin/env python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics

import numpy as np
import pickle
import cv2

import matplotlib.pyplot as plt

def motion_kinds(point_detect):
        x = []
        y = []
        for i in point_detect:
            x.append(i[0])
            y.append(i[1])
        point_detect = np.array(point_detect)
        
        detect = np.arange(15).reshape((15,1)).tolist()

        i_point_detect = point_detect[0]
        for iii in range(0,15):
            detect[int(iii)] =  [point_detect[int(iii)][0] - i_point_detect[0],
                            point_detect[int(iii)][1] - i_point_detect[1],
                            point_detect[int(iii)][2] - i_point_detect[2]]
        detect=np.array(detect)
        detect = np.reshape(detect,(1,45))

        a = loaded_model.predict(detect)

        plt.plot(x, y, color = 'black' , linewidth=2,
            label="dongtac %d" %a)
        plt.show()

        return a

def motion_detection(point_3d):
    global first_loop, cnt, fifteen_temporary_points, three_temporary_points
    if first_loop == True:
        #  original 15 point3d
        fifteen_temporary_points[cnt - 1] = [point_3d[0],
                            point_3d[1],
                            point_3d[2]]
        cnt+=1
        if cnt >= 15:
            first_loop = False
            cnt = 0
            return motion_kinds(fifteen_temporary_points)
    else:
        # print('test ne')
        # print('hieu x = \n',point_3d[0]-fifteen_temporary_points[14][0])
        # print('hieu y = \n',point_3d[1]-fifteen_temporary_points[14][1])
        # print('hieu z = \n',point_3d[2]-fifteen_temporary_points[14][2])
        
        if cnt <= 2:
                          
            three_temporary_points[cnt] = [point_3d[0],
                                point_3d[1],
                                point_3d[2]]
            cnt+=1

        else:
            for ii in range(0,12):
                fifteen_temporary_points[int(ii)] = [fifteen_temporary_points[int(ii)+3][0],
                                                    fifteen_temporary_points[int(ii)+3][1],
                                                    fifteen_temporary_points[int(ii)+3][2]]
                
            for ii in range(12,15):
                fifteen_temporary_points[int(ii)] = [three_temporary_points[int(ii)-12][0],
                                    three_temporary_points[int(ii)-12][1],
                                    three_temporary_points[int(ii)-12][2]]
            cnt = 0
            
        return motion_kinds(fifteen_temporary_points)

 
if __name__ == "__main__":
    try:
        cap1 = cv2.VideoCapture('/home/bao/Desktop/DATN/videosource/dongtac_xoaytron.avi')

        f1 = np.load("/home/bao/Desktop/DATN/npy_source/dongtac_xoaytron.npy")
        loaded_model = pickle.load(open('/home/bao/Desktop/DATN/sp/traindongtac.sav', 'rb'))

        first_loop = True
        three_temporary_points = np.arange(3).reshape((3,1)).tolist()
        fifteen_temporary_points = np.arange(15).reshape((15,1)).tolist()
        cnt = 0

        for index in range(int(cap1.get(7))):

            ret,frame = cap1.read()
            point_3d = f1[int(index)]
            if point_3d is not None:
                if first_loop == True:
                    k = motion_detection(point_3d)
                else:
                    if abs(point_3d[0]-fifteen_temporary_points[14][0]) <= 0.05 and abs(point_3d[1]-fifteen_temporary_points[14][1])<=0.05 and abs(point_3d[2]-fifteen_temporary_points[14][2])<=0.05:
                        continue
                    else:
                        k = motion_detection(point_3d)
                # print(k)
            cv2.imshow('video',frame)

            if cv2.waitKey(30) ==27 :
                break
    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))