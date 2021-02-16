#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from point_cloud2 import xyz_array_to_pointcloud2
import image_geometry
from cv_bridge import CvBridge

model = image_geometry.PinholeCameraModel()
bridge = CvBridge()

def callback(img):
    img = bridge.imgmsg_to_cv2(img, img.encoding)
    img = img / 100
    xyz = pixel2ray(img)
    points = xyz_array_to_pointcloud2(xyz, rospy.Time.now(), 'depth')
    pub.publish(points)

def callback_camera_info(cam_info):
    model.fromCameraInfo(cam_info)

def pixel2ray(depth):
    h, w = depth.shape
    depth = depth.reshape(-1)

    mesh = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    mesh = np.stack(mesh, axis=-1)
    mesh = mesh.reshape((-1, 2))

    x = (((mesh[:, 0] - model.cx()) * depth) - model.Tx()) / model.fx()
    y = (((mesh[:, 1] - model.cy()) * depth) - model.Ty()) / model.fy()
    z = depth

    xyz = np.stack([x, y, z], axis=-1)

    return xyz

if __name__ == '__main__':
    rospy.init_node('depth_to_points')
    
    depth_image_topic = rospy.get_param('~depth_image_topic', '/depth/image_raw')
    camera_info_topic = rospy.get_param('~camera_info_topic', '/depth/camera_info')
    point_topic = rospy.get_param('~point_topic', '/points_depth')

    sub1 = rospy.Subscriber(depth_image_topic, Image, callback, tcp_nodelay=True)
    sub2 = rospy.Subscriber(camera_info_topic, CameraInfo, callback_camera_info)
    pub = rospy.Publisher(point_topic, PointCloud2, queue_size=10)
    rospy.spin()
