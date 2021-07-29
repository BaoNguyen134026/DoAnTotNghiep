
import math as m
import time
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import open3d as o3d
import copy
import pandas as pd

aa = np.load("skeleton3D.npy")
# print(aa)
skeleton = []
for i in range(len(aa)):
    bb = aa[i,1]*-1
    cc = aa[i,2]*-1
    skeleton.append([aa[i,0], bb, cc])
skeleton.append([0,0,0])
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(skeleton)

o3d.visualization.draw_geometries([pc])