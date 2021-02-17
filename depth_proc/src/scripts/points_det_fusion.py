#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import tf2_ros

from sensor_msgs.msg import CameraInfo, PointCloud2
from nav_msgs.msg import OccupancyGrid
# from vision_msgs.msg import Detection2DArray
from autoware_msgs.msg import DetectedObjectArray
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from point_cloud2 import *

import time

det_msg = []
cam_info = None
no_ground = None
costmap = None

CLASS = {1: {'name': 'person', 'use': True, 'thresh': 0.5},  
         2: {'name': 'bicycle', 'use': True, 'thresh': 0.5}, 
         3: {'name': 'car', 'use': True, 'thresh': 2}, 
         4: {'name': 'truck', 'use': True, 'thresh': 4}, 
         5: {'name': 'sign', 'use': False, 'thresh': 0.5}, 
         6: {'name': 'sign', 'use': False, 'thresh': 0.5}, 
         7: {'name': 'sign', 'use': False, 'thresh': 0.5}, 
         8: {'name': 'triangle', 'use': True, 'thresh': 0.5}, 
         9: {'name': 'drum', 'use': True, 'thresh': 1}, 
         10: {'name': 'traffic_light', 'use': False, 'thresh': 0.5}}

def callback(msg):
    global det_msg
    global cam_info
    global no_ground
    global costmap

    if no_ground is not None:
        h, w = cam_info.height, cam_info.width
        mesh = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
        mesh = np.stack(mesh, axis=-1)
        mesh = mesh.reshape((-1, 2))

        points = pointcloud2_to_xyz_array(msg)
        det_list = []
        det_cls = []

        # cluster detected region
        for det in det_msg:
            if CLASS[det['cls']]['use'] is False:
                continue

            lt = det['bbox'][:2]
            rb = det['bbox'][2:]

            area = np.all((mesh > lt) & (mesh < rb), axis=1)
            points_area = points[area]

            dist = np.linalg.norm(points_area, axis=1)

            dom = np.bincount(dist.astype(int))
            dom = np.where(dom > 50)[0]

            if len(dom) == 0:
                dom = np.bincount(dist.astype(int))
                dom = np.where(dom > 25)[0]

            dom = dom[0]

            points_area = points_area[(dist > dom - CLASS[det['cls']]['thresh']) & (dist < dom + CLASS[det['cls']]['thresh'])]
            det_list.append(points_area)
            det_cls.append(det['cls'])

        # merge with points no ground


        try:
            trans = tf2_buffer.lookup_transform(no_ground.header.frame_id, msg.header.frame_id, rospy.Time(0))
        except:
            trans = None
            rospy.logwarn('TF not found')

        if trans is not None and len(det_list) > 0:
            det_points = np.vstack(det_list)
            det_points = xyz_array_to_pointcloud2(det_points, rospy.Time.now(), msg.header.frame_id)
            det_points = do_transform_cloud(det_points, trans)
            det_points = pointcloud2_to_xyz_array(det_points)

            ng_points = pointcloud2_to_xyz_array(no_ground)
            ng_points = np.concatenate([ng_points, det_points])
            ng_points = xyz_array_to_pointcloud2(ng_points, rospy.Time.now(), no_ground.header.frame_id)

            # map to occupancy grid
            if costmap is not None:
                w, h = costmap.info.width, costmap.info.height
                x, y, r = costmap.info.origin.position.x, costmap.info.origin.position.y, costmap.info.resolution
                
                grid = np.array(costmap.data).reshape((h, w))

                for pts, cls in zip(det_list, det_cls):
                    pts = xyz_array_to_pointcloud2(pts)
                    pts = do_transform_cloud(pts, trans)
                    pts = pointcloud2_to_xyz_array(pts)

                    min_pts = ((np.min(pts[:, :2], axis=0) - [x, y]) / r).astype(int) - 1
                    max_pts = ((np.max(pts[:, :2], axis=0) - [x, y]) / r).astype(int) + 1
                    grid[min_pts[1]:max_pts[1], min_pts[0]:max_pts[0]] = cls

                costmap.data = grid.reshape(-1).tolist()
            
        else:
            ng_points = no_ground

        pub1.publish(ng_points)
        pub2.publish(costmap)


def callback_camera_info(msg):
    global cam_info
    cam_info = msg

def callback_no_ground(msg):
    global no_ground
    no_ground = msg

def callback_costmap(msg):
    global costmap
    costmap = msg

def callback_det(msg):
    global det_msg
    dets = []
    for det in msg.objects:
        bbox = [det.x - det.width / 2, 
                det.y - det.height / 2, 
                det.x + det.width / 2, 
                det.y + det.height / 2]
        cls = det.id
        dets.append(dict(bbox=bbox, cls=cls))
    
    det_msg = dets

if __name__ == '__main__':
    rospy.init_node('points_det_fusion')

    tf2_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf2_buffer)
    
    det_topic = rospy.get_param('~det_topic', '/im_detector/objects')
    in_point_topic = rospy.get_param('~in_point_topic', '/points_depth')
    no_ground_topic = rospy.get_param('~no_ground_topic', '/points_no_ground')
    camera_info_topic = rospy.get_param('~camera_info_topic', '/depth/camera_info')
    out_point_topic = rospy.get_param('~out_point_topic', '/points_det')
    in_costmap_topic = rospy.get_param('~in_costmap_topic', '/semantics/costmap_generator/occupancy_grid')
    out_costmap_topic = rospy.get_param('~out_costmap_topic', '/aligned_grid')

    sub1 = rospy.Subscriber(in_point_topic, PointCloud2, callback, tcp_nodelay=True)
    sub2 = rospy.Subscriber(det_topic, DetectedObjectArray, callback_det)
    sub3 = rospy.Subscriber(camera_info_topic, CameraInfo, callback_camera_info)
    sub4 = rospy.Subscriber(no_ground_topic, PointCloud2, callback_no_ground)
    sub5 = rospy.Subscriber(in_costmap_topic, OccupancyGrid, callback_costmap)

    pub1 = rospy.Publisher(out_point_topic, PointCloud2, queue_size=10)
    pub2 = rospy.Publisher(out_costmap_topic, OccupancyGrid, queue_size=10)
    rospy.spin()
