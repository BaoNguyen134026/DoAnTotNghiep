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



a = np.arange(15).reshape((15,)).tolist()
print(a)
print(np.count_nonzero(a>5))