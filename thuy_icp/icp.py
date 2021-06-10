import open3d as o3d
import copy
import numpy as np
import math


# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])
    

# source = o3d.io.read_point_cloud("obj1.pcd")
# target = o3d.io.read_point_cloud("obj0.pcd")

threshold = 0.02

# trans_init = np.asarray([[1, 0, 0, 0],
#                          [0, 1, 0, 0],
#                          [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])

# print("Apply point-to-plane ICP")
# reg_p2l = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init, 
#     o3d.pipelines.registration.TransformationEstimationPointToPlane())
# # print(reg_p2l)
# print("Transformation is:")
# print(reg_p2l.transformation)

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

"""

# Mesh object
# radii = [0.001,0.0015 ,0.003,0.005]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     downpcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([rec_mesh])


# evaluate the confident
# print("Initial alignment")
# evaluation = o3d.pipelines.registration.evaluate_registration(
#     source, target, threshold, trans_init)
# print(evaluation)

source = o3d.io.read_point_cloud("obj1.pcd")
target = o3d.io.read_point_cloud("obj0.pcd")

threshold = 0.02

trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])

# trans_init = np.asarray([[ 0.42759956,  0.11147117, -0.897069,   -1.6197444 ],
#                          [-0.16436256,  0.98541353,  0.04410346,  0.11142216],
#                          [-0.16436256,  0.98541353,  0.04410346,  0.11142216],
#                          [ 0.0,         0.0,         0.0,         1.0]])

# print("\n Apply point-to-point ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)



# reg_p2p = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)


################################################################################
print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init, 
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
# print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
# print(trans_init)
# draw_registration_result(source, target, reg_p2l.transformation)


# source + target -> new point cloud (1 variable)
# source_temp = copy.deepcopy(source)
# target_temp = copy.deepcopy(target)
source.paint_uniform_color([1, 0.706, 0])
target.paint_uniform_color([0, 0.651, 0.929])
source.transform(reg_p2l.transformation)
# source.transform(trans_init)
newpointcloud = source + target
o3d.visualization.draw_geometries([newpointcloud])

# print("Downsample the point cloud with a voxel of 0.005")
# downpcd = newpointcloud.voxel_down_sample(voxel_size=0.005)
# o3d.visualization.draw_geometries([downpcd])

# Mesh object
# radii = [0.001,0.0015 ,0.003,0.005]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     downpcd, o3d.utility.DoubleVector(radii))
# mesh_in = rec_mesh
# o3d.visualization.draw_geometries([mesh_in])


# print('filter with Laplacian with 50 iterations')
# mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out]) """