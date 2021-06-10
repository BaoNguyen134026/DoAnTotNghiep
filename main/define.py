import math as m
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics

def motion_kinds(point_detect):
        point_detect = np.array(point_detect)
        detect = np.arange(len(point_detect)).reshape((len(point_detect),1)).tolist()
        P15_distance = np.arange(len(point_detect)).reshape((len(point_detect),0)).tolist()
        for i in range(1,len(point_detect)):
            P15_distance[i] = m.sqrt(m.pow(point_detect[i][0]-point_detect[i-1][0],2)
                                    +m.pow(point_detect[i][1]-point_detect[i-1][1],2)
                                    +m.pow(point_detect[i][2]-point_detect[i-1][2],2))
        b = [i for i in P15_distance if i >0.01]
        # print('len(b) =',len(b))
        if len(b) > 11:
            for i in range(0,15):
                detect[int(i)] =  [point_detect[int(i)][0] - point_detect[0][0],
                                    point_detect[int(i)][1] - point_detect[0][1],
                                    point_detect[int(i)][2] - point_detect[0][2]]
            detect=np.array(detect)
            detect = np.reshape(detect,(1,45))
            a = loaded_model.predict(detect)
        else: a = None
        return a 