#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
# import shapefile as sp
# import tqdm

from easydict import EasyDict as ed

from std_msgs.msg import ColorRGBA
from geodesy.utm import fromLatLong
from geometry_msgs.msg import PoseStamped, Vector3, Pose, Quaternion, Point
from sensor_msgs.msg import NavSatFix, Imu
from visualization_msgs.msg import MarkerArray, Marker
from hdmap_msgs.msg import HDmap
import rospy

LINK_TYPE = {'1': 'intersection',
             '2': 'tollgate_highpass',
             '3': 'tollgate',
             '4': 'bus_only',
             '5': 'switchable_way',
             '6': 'general',
             '7': 'to_rest_area',
             '8': 'in_rest_area',
             '9': 'out_rest_area',
             '10': 'to_rest_area',
             '11': 'in_rest_area',
             '12': 'out_rest_area',
             '99': 'etc'}

LINE_TYPE = {'111': {'type': Marker.LINE_STRIP,  'color': 'y', 'width': 0.3},
             '112': {'type': Marker.LINE_LIST, 'color': 'y', 'width': 0.3},
             '113': {'type': Marker.LINE_STRIP,  'color': 'y', 'width': 0.3},
             '114': {'type': Marker.LINE_STRIP,  'color': 'y', 'width': 0.3},
             '121': {'type': Marker.LINE_STRIP,  'color': 'y', 'width': 1.0},
             '122': {'type': Marker.LINE_LIST, 'color': 'y', 'width': 1.0},
             '123': {'type': Marker.LINE_STRIP,  'color': 'y', 'width': 1.0},
             '124': {'type': Marker.LINE_STRIP,  'color': 'y', 'width': 1.0},

             '211': {'type': Marker.LINE_STRIP,  'color': 'w', 'width': 0.3},
             '212': {'type': Marker.LINE_LIST, 'color': 'w', 'width': 0.3},
             '213': {'type': Marker.LINE_STRIP,  'color': 'w', 'width': 0.3},
             '214': {'type': Marker.LINE_STRIP,  'color': 'w', 'width': 0.3},
             '221': {'type': Marker.LINE_STRIP,  'color': 'w', 'width': 1.0},
             '222': {'type': Marker.LINE_LIST, 'color': 'w', 'width': 1.0},
             '223': {'type': Marker.LINE_STRIP,  'color': 'w', 'width': 1.0},
             '224': {'type': Marker.LINE_STRIP,  'color': 'w', 'width': 1.0},

             '311': {'type': Marker.LINE_STRIP,  'color': 'b', 'width': 0.3},
             '312': {'type': Marker.LINE_LIST, 'color': 'b', 'width': 0.3},
             '313': {'type': Marker.LINE_STRIP,  'color': 'b', 'width': 0.3},
             '314': {'type': Marker.LINE_STRIP,  'color': 'b', 'width': 0.3},
             '321': {'type': Marker.LINE_STRIP,  'color': 'b', 'width': 1.0},
             '322': {'type': Marker.LINE_LIST, 'color': 'b', 'width': 1.0},
             '323': {'type': Marker.LINE_STRIP,  'color': 'b', 'width': 1.0},
             '324': {'type': Marker.LINE_STRIP,  'color': 'b', 'width': 1.0},

             '999': {'type': Marker.LINE_STRIP,  'color': 'g', 'width': 0.3},
             '': {'type': Marker.LINE_STRIP,  'color': 'g', 'width': 0.3},
             }

LINE_KIND = {'501': 'center_line',
             '5011': 'switchable_lane',
             '502': 'u_turn',
             '503': 'line',
             '504': 'bus_only',
             '505': 'edge',
             '506': 'no_lane_change',
             '515': 'no_parking',
             '525': 'guide_line',
             '530': 'stop_line',
             '531': 'safe_area',
             '535': 'bicycle_road',
             '599': 'etc'}

class hdmap_loader:
    def __init__(self, map_dir, type, center=None):
        self.map_dir = map_dir
        self.type = type

        center = fromLatLong(*center)
        self.center = center

        self.read_map()
        self.hdmap_vis()

    def read_map(self):
        pkl_file = os.path.join(self.map_dir, self.type + '.pkl')

        self.shaperecords = []

        if os.path.isfile(pkl_file) is True:
            self.load_map_data(pkl_file)
            rospy.logdebug('pre loaded map found')

        self.shaperecords = [ed(i) for i in self.shaperecords]

        rospy.logdebug('hdmap data loaded')

    def save_map_data(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.shaperecords, f)

    def load_map_data(self, pkl_file):
        with open(pkl_file, 'rb') as f:
             self.shaperecords = pickle.load(f)
    
    def make_marker(self, shaperecord, id, color=None):
        shape = shaperecord.shape
        record = shaperecord.record

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        
        marker.id = id
        marker.action = Marker.ADD

        if color is None:
            if self.type == 'link':
                marker.type = Marker.LINE_STRIP
                marker.scale.x = 0.3
                marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)

            elif self.type == 'surfacelinemark':
                line_type = LINE_TYPE[record.Type]
                marker.type = line_type['type']
                marker.scale.x = line_type['width']
                marker.color = ColorRGBA(*((1.0, 1.0, 1.0, 1.0) if line_type['color'] == 'w' else
                                        (1.0, 1.0, 0.0, 1.0) if line_type['color'] == 'y' else
                                        (0.0, 0.0, 1.0, 1.0) if line_type['color'] == 'b' else
                                        (0.8, 0.6, 0.8, 1.0)))
            else:
                marker.type = Marker.LINE_LIST
                marker.scale.x = 0.1
                marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.5)
        else:
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 1.0
            marker.color = ColorRGBA(*color)

        point_list = []
        for point in shape.points:
            point_list.append(Point(point[0], point[1], point[2]))

        if marker.type == Marker.LINE_LIST and len(point_list) %2 !=0:
            point_list.append(Point(point[0], point[1], point[2]))
        marker.points = point_list
        marker.pose.position = Point(0, 0, 0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        return marker

    def make_markerarray(self):
        markerarray = []
        id = 0
        for shaperecord in self.shaperecords:
            markerarray.append(self.make_marker(shaperecord, id))
            id += 1
        return MarkerArray(markerarray)

    def hdmap_vis(self):
        self.pub = rospy.Publisher("/hdmap/" + self.type, MarkerArray, queue_size=10, latch=True)
        self.pub.publish(self.make_markerarray())

if __name__ == "__main__":
    import time
    rospy.init_node('map_vis', anonymous=True)
    root_dir = os.path.join(os.path.split(__file__)[0], './map_data')  

    rospy.spin()
