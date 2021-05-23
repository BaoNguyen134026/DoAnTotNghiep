#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import time
# import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics

# import numpy as np
import pickle
import cv2

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
        else:
            save = None
            return save
def motion_kinds(point_detect):
        point_detect = np.array(point_detect)
        detect = np.arange(15).reshape((15,1)).tolist()

        i_point_detect = point_detect[0]
        for iii in range(0,15):
            detect[int(iii)] =  [point_detect[int(iii)][0] - i_point_detect[0],
                            point_detect[int(iii)][1] - i_point_detect[1],
                            point_detect[int(iii)][2] - i_point_detect[2]]
        detect=np.array(detect)
        detect = np.reshape(detect,(1,45))

        a = loaded_model.predict(detect)
        # print(a)
        return a
def motion_detection(point_3d):
    global first_loop, cnt, fifteen_temporary_points, three_temporary_points
    if first_loop == True:
        fifteen_temporary_points[cnt - 1] = [point_3d[0],
                            point_3d[1],
                            point_3d[2]]
        cnt+=1
        if cnt >= 15:
            first_loop = False
            cnt = 0
            return motion_kinds(fifteen_temporary_points)

    else:
        if point_3d[0] - fifteen_temporary_points[0] >= 0.5:
            
            pass
        if cnt <= 2:
            three_temporary_points[cnt] = [point_3d[0],
                                point_3d[1],
                                point_3d[2]]
            cnt+=1
        else:
            for ii in range(0,12):
                fifteen_temporary_points[int(ii)] = [fifteen_temporary_points[int(ii)+3][0],
                                                    fifteen_temporary_points[int(ii)+3][1],
                                                    fifteen_temporary_points[int(ii)+3][2]]
            for ii in range(12,15):
                fifteen_temporary_points[int(ii)] = [three_temporary_points[int(ii)-12][0],
                                    three_temporary_points[int(ii)-12][1],
                                    three_temporary_points[int(ii)-12][2]]
            cnt = 0
        return motion_kinds(fifteen_temporary_points)
# Main content begins
if __name__ == "__main__":
    try:
        # Configure depth and color streams of the intel realsense
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        # Start the realsense pipeline
        pipeline = rs.pipeline()
        pipeline.start(config)
        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)
        # Get the intrinsics information for calculation of 3D point
        unaligned_frames = pipeline.wait_for_frames()
        frames = align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
        color = frames.get_color_frame()
        color_image = np.asanyarray(color.get_data())
        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2
        # Create window for initialisation
        window_name = "cubemos skeleton tracking with realsense D400 series"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        out = cv2.VideoWriter('body_yen_tudo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (color_image.shape[1],color_image.shape[0]))
        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (848,480))
        first_loop = True
        three_temporary_points = np.arange(3).reshape((3,1)).tolist()
        fifteen_temporary_points = np.arange(15).reshape((15,1)).tolist()
        cnt = 0
        loaded_model = pickle.load(open('sp/traindongtac.sav', 'rb'))
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            unaligned_frames = pipeline.wait_for_frames()
            frames = align.process(unaligned_frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth.get_data())
            color_image = np.asanyarray(color.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(color_image)
            # print(skeletons)
            # render the skeletons on top of the acquired image and display it
            cm.render_result(skeletons, color_image, joint_confidence)
            points3d_skeleton = render_ids_3d(color_image, skeletons, depth, depth_intrinsic, joint_confidence)

            if points3d_skeleton is not None:
                montion_kind = motion_detection(points3d_skeleton[4])
                print(montion_kind)
                        
            cv2.imshow(window_name, color_image)
            if cv2.waitKey(1) == 27:
                break

        pipeline.stop()
        cv2.destroyAllWindows()
        # matrix = np.array(matrix)
        # print('matrix = ',matrix)
        # np.save('body_yen_tudo',matrix)

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
