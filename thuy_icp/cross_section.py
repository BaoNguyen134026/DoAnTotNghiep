# import numpy as np
# # import open3d as o3d
# import copy

# # take  point cloud
# pcd = o3d.io.read_point_cloud('obj0.pcd')

# ##################################################################################
# xyz = np.asarray(pcd.points)
# xyz1=[]
# for i in range(len(xyz[:,0])):
#     if xyz[i,1] < -1.76 or xyz[i,1] > -1.75:
#         continue    
#     else:
#         xyz1.append([xyz[i,0],xyz[i,1],xyz[i,2]])
# print(xyz1)

# for i in range(len(xyz1)):
#     xyz1[i][1] = 0

# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(xyz1)
# o3d.visualization.draw_geometries([pc])
# # o3d.io.write_point_cloud("cross.pcd", pc)

