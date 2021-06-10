import open3d as o3d
import copy
import numpy as np
import math

threshold = 0.02
source = o3d.io.read_point_cloud("source.pcd")
target = o3d.io.read_point_cloud("target.pcd")


trans_init = np.asarray([[ 0.42759956,  0.11147117, -0.897069,   -1.6197444 ],
                         [-0.16436256,  0.98541353,  0.04410346,  0.11142216],
                         [ 0.8889002,   0.12858594,  0.43968408, -1.25518773],
                         [ 0.0,         0.0,         0.0,         1.0]])

source.paint_uniform_color([1, 0.706, 0])
target.paint_uniform_color([0, 0.651, 0.929])
# source.transform(reg_p2l.transformation)
source.transform(trans_init)

o3d.visualization.draw_geometries([source, target])

newpointcloud = source + target
print("Downsample the point cloud with a voxel of 0.05")
downpcd = newpointcloud.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([downpcd])

xyz = np.asarray(downpcd.points)
print(xyz.shape)
R = 0.05
xyz_filter=[]
for i in range(len(xyz[:,0])):
    c = 0
    for j in range(len(xyz[:,0])):
        if xyz[j,0]<(xyz[i,0]+R) and xyz[j,0]>(xyz[i,0]-R) and xyz[j,1]<(xyz[i,1]+R) and xyz[j,1]>(xyz[i,1]-R) and xyz[j,2]<(xyz[i,2]+R) and xyz[j,2]>(xyz[i,2]-R):
            c += 1
        else: 
            continue
    if c > 20:
        xyz_filter.append([xyz[i,0],xyz[i,1],xyz[i,2]])
    else:
        continue
print(len(xyz_filter))
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(xyz_filter)
o3d.visualization.draw_geometries([pc])
o3d.io.write_point_cloud("pc.pcd", pc)