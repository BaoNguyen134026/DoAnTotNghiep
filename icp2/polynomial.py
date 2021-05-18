import cv2
import open3d as o3d
import copy
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# take  point cloud
pcd = o3d.io.read_point_cloud('obj2.pcd')
x=[]
y=[]
xyz = np.asarray(pcd.points)
xyz1=[]
for i in range(len(xyz[:,0])):
    if xyz[i,1] < -1.6 or xyz[i,1] > -1.59:
        continue    
    else:
        xyz1.append([xyz[i,0],xyz[i,1],xyz[i,2]])
        # x.append(xyz[i][0])
        # y.append(xyz[i][2])
# print(len(xyz1))
for j in range(20):
    # print(j)
    i = randint(0,len(xyz1)-1)
    # print(i)
    # xyz1[i][1] = 0
    x.append(xyz1[i][0])
    y.append(xyz1[i][2])

print(len(x))
# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(xyz1)
# o3d.visualization.draw_geometries([pc])


# print(__doc__)



# Author: Mathieu Blondel
#         Jake Vanderplas
# License: BSD 3 clause



# def f(x):
#     """ function to approximate by polynomial interpolation"""
#     return x * np.sin(x)


# generate points used to plot
# x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
# x = np.linspace(0, 10, 100)
# rng = np.random.RandomState(0)
# rng.shuffle(x)
# x = np.sort(x[:20])
# y = f(x)

x=np.array(x)
y=np.array(y)
# x = np.sort(x[:20])
# y = np.sort(x[:20])

# print(y)

# create matrix versions of these arrays
X = x[:, np.newaxis]
# X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
# plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
#          label="ground truth")
# plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")


# for count, degree in enumerate([3, 4, 5]):
#     model = make_pipeline(PolynomialFeatures(degree), Ridge())
#     mot.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
#              label="degree %d" % degree)
# print(X)
model = make_pipeline(PolynomialFeatures(4), Ridge())
model.fit(X, y)
y_plot = model.predict(X)
# print(X.shape)
plt.plot(x, y_plot, color = 'yellowgreen' , linewidth=lw,
            label="degree %d" % 2)
plt.legend(loc='lower left')
# plt.plot(X, y_plot, color = 'yellowgreen' , linewidth=lw,
#             label="degree %d" % 2)
# plt.legend(loc='lower left')
plt.show()
# del.fit(X, y)
#     y_plot = model.predict(X)
#     pl