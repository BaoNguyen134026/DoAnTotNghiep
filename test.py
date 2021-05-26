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
import math as m
import matplotlib.pyplot as plt



a = np.arange(15*3).reshape((15,3))
print('mang truoc:\n',a)
a[0][1] = 0.002-0.004
print('mang sau: \n',a)
# b = [i for i in a if i >5]
# print('b = ',b)
# print('len b = ',len(b))
# enumerate   
# print(a.count(1))