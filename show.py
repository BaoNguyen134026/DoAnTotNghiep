import open3d as o3d
import copy
import numpy as np
import math


pc = o3d.io.read_point_cloud("pc_thuy.pcd")
o3d.visualization.draw_geometries([pc])
