
import math as m
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import copy

pipeline = rs.pipeline()
config = rs.config()
colorizer = rs.colorizer()

# colorized = colorizer.process(frames)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

pipeline.start(config)



decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude,2)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude,2)
spatial.set_option(rs.option.filter_smooth_alpha,0.5)
spatial.set_option(rs.option.filter_smooth_delta,20)

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha,0.4)
temporal.set_option(rs.option.filter_smooth_delta,20)

filters = [rs.disparity_transform(),
           rs.spatial_filter(),
           rs.temporal_filter(),
           rs.disparity_transform(False)]

for i in range(30):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

for f in filters:
    depth_frame = f.process(depth_frame)

colorized = colorizer.process(frames)

# profile = pipeline.get_active_profile()
# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# depth_intrinsics = depth_profile.get_intrinsics()

pipeline.stop()

###########get file .ply
ply = rs.save_to_ply("test.ply")
ply.set_option(rs.save_to_ply.option_ply_binary, False)
ply.set_option(rs.save_to_ply.option_ply_normals, True)
ply.process(colorized)

pcd = o3d.io.read_point_cloud('test.ply')
o3d.visualization.draw_geometries([pcd])