import cv2
from matplotlib import interactive
from numpy.core.fromnumeric import reshape
import open3d as o3d
import copy
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# take  point cloud
pcd = o3d.io.read_point_cloud('pc.pcd')
xyz = np.asarray(pcd.points)
xyz1=[]
xy =[]
j = 0
for i in range(len(xyz[:,0])):
    if xyz[i,1] < -0.8 or xyz[i,1] > -0.785:
        continue 
    else:
        j += 1
        # print(j)
        xyz1.append([xyz[i,0],xyz[i,1],xyz[i,2]])
        xy.append([xyz[i][0],xyz[i][2]])


# tao key dat diem de sorted
def func(xy):
    return xy[0]
#lay 20 diem trong xy  va sap xep theo thu tu tu lon den nho
xy = sorted(xy[:20], key= func)
# lay x y  ra de tim ham`
x = []
y = []
for i in xy:
    x.append([i[0]])
    y.append([i[1]])

poly = PolynomialFeatures(degree = 10)
X_poly = poly.fit_transform(x) # x->poly
poly.fit(X_poly, y) # y->poly co x

lin2 = LinearRegression()
lin2.fit(X_poly, y)

plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
  
plt.show()