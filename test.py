
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

    skeleton.append([aa[i,0], -aa[i,1], -aa[i,2]])
skeleton.append([0,0,0])
pc_k = o3d.geometry.PointCloud()
pc_k.points = o3d.utility.Vector3dVector(skeleton)
pc_k.paint_uniform_color([1, 0, 0])

# take  point cloud
pcd = o3d.io.read_point_cloud('bao_and_skeleton.ply')

# down sample
pcd = pcd.voxel_down_sample(voxel_size=0.001)
# pc_r = copy.deepcopy(pcd)
# Rotation
# R = pcd.get_rotation_matrix_from_xyz((np.pi/20, 0, np.pi/2))
# # R = pcd.get_rotation_matrix_from_xyz((-np.pi/10,-np.pi/20,np.pi/2))
# pcd.rotate(R, center=(0, 0, 0))
# o3d.visualization.draw_geometries([pcd])


# ### Load json file and crop
# vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
# obj = vol.crop_point_cloud(pcd)
# # print(obj)
# o3d.visualization.draw_geometries([obj])
# o3d.io.write_point_cloud("obj2.pcd", obj)
xyz = np.asarray(pcd.points)
xyz1=[]
for i in range(len(xyz[:,0])):
    if xyz[i,0] < -0.7 or xyz[i,0] > 1 or xyz[i,1] < -2 or xyz[i,1] > 1 or xyz[i,2] < -2.25 or xyz[i,2] > -0.1 :
        continue    
    else:
        xyz1.append([xyz[i,0],xyz[i,1],xyz[i,2]])
# print(xyz1)
xyz1.append([0,0,0])
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(xyz1)
# o3d.visualization.draw_geometries([pc])
# o3d.io.write_point_cloud("obj2.pcd", pc)
# ### Find plane
pcd_tree = o3d.geometry.KDTreeFlann(pc)
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[len(xyz1)], 0.2)
np.asarray(pc.colors)[idx[1:], :] = [0, 1, 0]

obj = pc
plane_model, inliers = obj.segment_plane(distance_threshold=0.03,
                                         ransac_n=3,
                                         num_iterations=1000)
inlier_cloud = obj.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])

outlier_cloud = obj.select_by_index(inliers, invert=True)
outlier_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
outlier_cloud.paint_uniform_color([1.0, 0, 0])
pcd.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries([pc,pc_k])
o3d.io.write_point_cloud("hinh/obj2.pcd", outlier_cloud)

# # Mesh object
# radii = [0.01,0.015 ,0.03,0.05]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     outlier_cloud, o3d.utility.DoubleVector(radii))
# # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# # alpha = 0.09
# # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(outlier_cloud, alpha)
# # rec_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([rec_mesh],mesh_show_back_face=True)

# # print(rec_mesh)