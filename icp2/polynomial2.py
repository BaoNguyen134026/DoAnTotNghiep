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

# print(x)
# xy1=[xy]
# for j in range(len(x)):
#     xy1.append([x[j],y[j]])
# print("xy1: \n",xy1)
# # pc_xy1 = o3d.geometry.PointCloud()
# # pc_xy1.points = o3d.utility.Vector3dVector(xy1)
# # o3d.visualization.draw_geometries([pc_xy1])

"""
colors = ['teal', 'black', 'gold','red','blue','green']
#tim ham`
for i , degree in enumerate([2, 3, 4 ,5, 6,7]):
    model = make_pipeline(PolynomialFeatures(i), Ridge())
    model.fit(x, y)
    y_plot = model.predict(x)
    plt.plot(x, y_plot, color = colors[i] , linewidth=2,
                label="degree %d" % degree)
plt.plot(x, y, color = 'black' , linewidth=2,
            label="point get func")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

plt.legend(loc='lower left')

plt.show()
"""

"""
# https://www.youtube.com/watch?v=4b7ujubQ4jw

degree = 2
def PolynomialRefression(degree):
    regressor = LinearRegression()
    regressor.fit(x,y)
    xx = np.linspace(0, 2500, 3000)
    yy = regressor.predict(xx.reshape(xx.shape[0],1))

    quadratic_featurizer = PolynomialFeatures(degree)
    x_quadratic = quadratic_featurizer.fit_transform(x)

    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(x_quadratic,y)
    xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))

    print("Residual sum of squares : %.2f" % np.mean((regressor_quadratic.predict(x_quadratic)-y) ** 2))

    plt.plot(xx,yy)
    plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c = 'r',linestyle='--')
    plt.title('aa')
    plt.xlabel('bb')
    plt.ylabel('cc')
    plt.axis([-5000, 5000,-5000, 5000])
    plt.grid(True)
    plt.scatter(x,y)
    plt.show()

PolynomialRefression(4)
# i = interactive(PolynomialRefression, degree=(0,10))
"""

"""
# https://www.youtube.com/watch?v=3L_-JbFxftM

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
LinReg = LinearRegression()
LinReg.fit(x_train, y_train)

plt.scatter(x_train, y_train, color = 'green')
plt.plot(x_train, LinReg.predict(x_train), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""


# https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/

# lin = LinearRegression() 
# lin.fit(x, y)

# plt.scatter(x, y, color = 'blue')
# plt.plot(x, lin.predict(x), color = 'red')
# plt.title('Linear Regression')
# plt.xlabel('Temperature')
# plt.ylabel('Pressure')
# plt.show()

poly = PolynomialFeatures(degree = 10)
X_poly = poly.fit_transform(x) # x->poly
poly.fit(X_poly, y) # y->poly co x

lin2 = LinearRegression()
lin2.fit(X_poly, y)

# plt.scatter(x, y, color = 'blue')
  
plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
  
plt.show()