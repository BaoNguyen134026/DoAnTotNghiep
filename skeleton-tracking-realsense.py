#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker
count_frame =0
save = []
aaa = 0
# t = np.array([0 0 0])
def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence):
    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]

    # print('shape', render_image.shape)
    # print('channel',channel)
    distance_kernel_size = 5
    # if skeletons_2d == [] :
    #             save.append([0,0,0])
    #             np.save('save3d2',save)
    #             print('khong 4')
    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons_2d)):
        if skeletons_2d[skeleton_index] == []:
            break
        # print(len(skeletons_2d))
        # print('co s \n',skeletons_2d)
        skeleton_2D = skeletons_2d[skeleton_index]
        # print('khong s \n \r',skeleton_2D)

        joints_2D = skeleton_2D.joints
        # print('joints_2D',joints_2D)
        did_once = False
        i=0
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
                    # print(point_3d[0])
                    # save.append(point_3d)

                    # print(point_3d)
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    

                    point_str = [str(x) for x in point_3d]
                    # print(type(joint_index))
                    i = "{}".format(joint_index)
                    # print(save)
                    if joint_index == 4:
                        save.append(point_3d)
                        out.write(color_image)
                        np.save('sp/dongtac_im',save)
                        # aaa+=1
                        # print(aaa)
                    
                    cv2.putText(
                        render_image,
                    
                        str(i) + ' ' + str(point_3d) ,
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )
                # elif joint_index == 4:
                #     save.append([0,0,0])
                #     np.save('save3d2',save)
                #     print('khong 4')
            # elif joint_index == 4 and skeletons_2d != []:
            #     save.append([0,0,0])
            #     np.save('save3d2',save)
            #     print('khong 4')





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
        out = cv2.VideoWriter('sp/dongtac_im.mp4',cv2.VideoWriter_fourcc('MP4V'), 30, (color_image.shape[1],color_image.shape[0]))
        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (848,480))

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

            # print(color_image.shape)

            
            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(color_image)
            # print(skeletons)
            # render the skeletons on top of the acquired image and display it
            cm.render_result(skeletons, color_image, joint_confidence)
            
            render_ids_3d(
                color_image, skeletons, depth, depth_intrinsic, joint_confidence
            )
            # count_frame+=1
            # out.write(color_image)


            cv2.imshow(window_name, color_image)
            if cv2.waitKey(1) == 27:
                break

        pipeline.stop()
        cv2.destroyAllWindows()

    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))
