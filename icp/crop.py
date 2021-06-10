
import math as m
import time
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import open3d as o3d
import copy
import pandas as pd

"""
# take  point cloud
pcd = o3d.io.read_point_cloud('test2.ply')

# down sample
pcd = pcd.voxel_down_sample(voxel_size=0.001)
# pc_r = copy.deepcopy(pcd)

##################################################################################
# Rotation 
R = pcd.get_rotation_matrix_from_xyz((np.pi/20, 0, np.pi/2))
# R = pcd.get_rotation_matrix_from_xyz((-np.pi/10,-np.pi/20,np.pi/2))
pcd.rotate(R, center=(0, 0, 0))
# o3d.visualization.draw_geometries([pcd])


##################################################################################
# ### Load json file and crop
# vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
# obj = vol.crop_point_cloud(pcd)
# # print(obj)
# o3d.visualization.draw_geometries([obj])
# o3d.io.write_point_cloud("obj2.pcd", obj)


##################################################################################
xyz = np.asarray(pcd.points)
xyz1=[]
for i in range(len(xyz[:,0])):
    if xyz[i,0] < -0.2 or xyz[i,0] > 0.8 or xyz[i,1] < -2 or xyz[i,1] > 1 or xyz[i,2] < -2.25 or xyz[i,2] > -1 :
        continue    
    else:
        xyz1.append([xyz[i,0],xyz[i,1],xyz[i,2]])
print(len(xyz1))

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(xyz1)
# o3d.visualization.draw_geometries([pc])
# o3d.io.write_point_cloud("obj2.pcd", pc)


##################################################################################
# Rotation
R = pc.get_rotation_matrix_from_xyz((-np.pi/6, 0, 0))
# R = pcd.get_rotation_matrix_from_xyz((-np.pi/10,-np.pi/20,np.pi/2))
pc.rotate(R, center=(0, 0, 0))
# o3d.visualization.draw_geometries([pc])


##################################################################################
# ### Find plane
obj = pc
plane_model, inliers = obj.segment_plane(distance_threshold=0.03,
                                         ransac_n=3,
                                         num_iterations=1000)
inlier_cloud = obj.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])

outlier_cloud = obj.select_by_index(inliers, invert=True)
outlier_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([outlier_cloud])
o3d.io.write_point_cloud("obj2.pcd", outlier_cloud)
"""


source1 = o3d.io.read_point_cloud("obj1.pcd")
target1 = o3d.io.read_point_cloud("obj0.pcd")

xyz = np.asarray(source1.points)
source_temp=[]
for i in range(len(xyz[:,0])):
    if xyz[i,0] < -0.1 or xyz[i,0] > 0.4:
        continue    
    else:
        source_temp.append([xyz[i,0],xyz[i,1],xyz[i,2]])
pc_source = o3d.geometry.PointCloud()
pc_source.points = o3d.utility.Vector3dVector(source_temp)
o3d.visualization.draw_geometries([pc_source])
o3d.io.write_point_cloud("source.pcd", pc_source)

xyz = np.asarray(target1.points)
target_temp=[]
for i in range(len(xyz[:,0])):
    if xyz[i,0] < 0.2 or xyz[i,0] > 1:
        continue    
    else:
        target_temp.append([xyz[i,0],xyz[i,1],xyz[i,2]])
pc_target = o3d.geometry.PointCloud()
pc_target.points = o3d.utility.Vector3dVector(target_temp)
o3d.visualization.draw_geometries([pc_target])
o3d.io.write_point_cloud("target.pcd", pc_target)