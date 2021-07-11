#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics
import pickle
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence):
    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]

    distance_kernel_size = 5
    
    for skeleton_index in range(len(skeletons_2d)):
        if skeletons_2d[skeleton_index] == []:
            break
        
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        did_once = False
        i=0
        cnt = 0
        save = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for joint_index in range(len(joints_2D)):
            if did_once == False:
                cv2.putText(
                    render_image,
                    "id: " + str(skeleton_2D.id),
                    (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    thickness,
                )
                did_once = True
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                
                distance_in_kernel = []
                low_bound_x = max(
                    0,
                    int(
                        joints_2D[joint_index].x - math.floor(distance_kernel_size / 2)
                    )
                )

                upper_bound_x = min(
                    cols - 1,
                    int(joints_2D[joint_index].x + math.ceil(distance_kernel_size / 2)),
                )

                low_bound_y = max(
                    0,
                    int(
                        joints_2D[joint_index].y - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_y = min(
                    rows - 1,
                    int(joints_2D[joint_index].y + math.ceil(distance_kernel_size / 2)),
                )
                for x in range(low_bound_x, upper_bound_x):
                    for y in range(low_bound_y, upper_bound_y):
                        distance_in_kernel.append(depth_map.get_distance(x, y))
                median_distance = np.percentile(np.array(distance_in_kernel), 50)
                depth_pixel = [
                    int(joints_2D[joint_index].x),
                    int(joints_2D[joint_index].y),
                ]
                if median_distance >= 0.3:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic, depth_pixel, median_distance
                    )
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    

                    point_str = [str(x) for x in point_3d]
                    i = "{}".format(joint_index)
                    
                    cnt +=1
                    save[cnt-1] = [point_3d[0],
                                        point_3d[1],
                                        point_3d[2]]
                    
                    cv2.putText(
                        render_image,
                    
                        str(i) + ' ' + str(point_3d) ,
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )
            
        if cnt == 18:
            return save
        else: return None
def post_process_depth_frame(depth_frame):
    """
    Filter the depth frame acquired using the Intel RealSense device

    Parameters:
    -----------
    depth_frame          : rs.frame()
                           The depth frame to be post-processed
    decimation_magnitude : double
                           The magnitude of the decimation filter
    spatial_magnitude    : double
                           The magnitude of the spatial filter
    spatial_smooth_alpha : double
                           The alpha value for spatial filter based smoothening
    spatial_smooth_delta : double
                           The delta value for spatial filter based smoothening
    temporal_smooth_alpha: double
                           The alpha value for temporal filter based smoothening
    temporal_smooth_delta: double
                           The delta value for temporal filter based smoothening

    Return:
    ----------
    filtered_frame : rs.frame()
                     The post-processed depth frame
    """
    
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_magnitude=1.0
    spatial_magnitude=2.0
    spatial_smooth_alpha=0.5
    spatial_smooth_delta=20
    temporal_smooth_alpha=0.4
    temporal_smooth_delta=20
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)
    return filtered_frame

def point_cloud(df1,df2):
    #Get pcd
    # Point Cloud
    pc = rs.pointcloud()
    pcl_1 = o3d.geometry.PointCloud()
    pcl_2 = o3d.geometry.PointCloud()

    points_1 = pc.calculate(df1)
    points_2 = pc.calculate(df2)
    v1 = points_1.get_vertices()
    v2 = points_2.get_vertices()
    verts_1 = np.asanyarray(v1).view(np.float32).reshape(-1, 3)  # xyz
    verts_2 = np.asanyarray(v2).view(np.float32).reshape(-1, 3)  # xyz
    pcl_1.points = o3d.utility.Vector3dVector(verts_1)
    pcl_2.points = o3d.utility.Vector3dVector(verts_2)
    # pcl = pcl.voxel_down_sample(voxel_size=0.017)
    # pcl = pcl.voxel_down_sample(voxel_size=0.017)
# Main content begins
if __name__ == "__main__":
    try:
        # Configure depth and color streams of the intel realsense
        #...from Camera 1
        config_1 = rs.config()
        config_1.enable_device('046122251324')
        config_1.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config_1.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        #...from Camera 2
        config_2 = rs.config()
        config_2.enable_device('108222250284')
        config_2.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config_2.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

        # Start the realsense pipeline
        #...from Camera 1
        pipeline_1 = rs.pipeline()
        pipeline_1.start(config_1)
        #...from Camera 2
        pipeline_2 = rs.pipeline()
        pipeline_2.start(config_2)

        # Create align object to align depth frames to color frames
        #...from Camera 1
        align_1 = rs.align(rs.stream.color)
        #...from Camera 2
        align_2 = rs.align(rs.stream.color)

        # Get the intrinsics information for calculation of 3D point
        #...from Camera 1
        unaligned_frames_1 = pipeline_1.wait_for_frames()
        frames_1 = align_1.process(unaligned_frames_1)
        depth_frame_1 = frames_1.get_depth_frame()
        depth_intrinsic_1 = depth_frame_1.profile.as_video_stream_profile().intrinsics
        color_1 = frames_1.get_color_frame()
        color_image_1= np.asanyarray(color_1.get_data())
        #...from Camera 2
        unaligned_frames_2 = pipeline_2.wait_for_frames()
        frames_2 = align_2.process(unaligned_frames_2)
        depth_frame_2 = frames_2.get_depth_frame()
        depth_intrinsic_2 = depth_frame_2.profile.as_video_stream_profile().intrinsics
        color_2 = frames_2.get_color_frame()
        color_image_2 = np.asanyarray(color_2.get_data())
        
        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2

        # Create window for initialisation
        window_name = "cubemos skeleton tracking with realsense D400 series"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

        # Initialize
        out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (color_image_1.shape[1],color_image_1.shape[0]))
        # loaded_model = pickle.load(open('byt.sav', 'rb'))
        P15_distance = np.arange(15).reshape((15,)).tolist()
        Points_15 = np.arange(15).reshape((15,1)).tolist()
        Points_3 = np.arange(3).reshape((3,1)).tolist()
        cnt = 0
        first_loop = True
        while True:
            # Create a pipeline_1 object. This object configures the streaming camera and owns it's handle
            #...from Camera 1
            unaligned_frames_1 = pipeline_1.wait_for_frames()
            frames_1 = align_1.process(unaligned_frames_1)
            depth_frame_1 = frames_1.get_depth_frame()
            color_1 = frames_1.get_color_frame()
            #...from Camera 2
            unaligned_frames_2 = pipeline_2.wait_for_frames()
            frames_2 = align_2.process(unaligned_frames_2)
            depth_frame_2 = frames_2.get_depth_frame()
            color_2 = frames_2.get_color_frame()

            if not depth_frame_1 or not depth_frame_2 or not color_1 or not color_2:
                continue
            # Convert images to numpy arrays
            #...from camera 1
            depth_image_1 = np.asanyarray(depth_frame_1.get_data())
            color_image_1 = np.asanyarray(color_1.get_data())
            color_image_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB)
            #...from camera 2
            depth_image_2 = np.asanyarray(depth_frame_2.get_data())
            color_image_2 = np.asanyarray(color_2.get_data())
            color_image_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2RGB)
            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(color_image_1)

            # render the skeletons on top of the acquired image and display it
            cm.render_result(skeletons, color_image_1, joint_confidence)
            P3d_Skeletons = render_ids_3d(  color_image_1,
                                        skeletons,
                                        depth_frame_1,
                                        depth_intrinsic_1,
                                        joint_confidence)
            # if not P3d_Skeletons:
            #     pass
            cv2.imshow(window_name, color_image_1)
            cv2.imshow('2',color_image_2)
            if cv2.waitKey(1) == 27:
                break
        pipeline_1.stop()
        cv2.destroyAllWindows()

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
