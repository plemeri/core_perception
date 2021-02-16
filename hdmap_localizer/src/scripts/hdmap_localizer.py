#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from logging import currentframe

from numpy import linalg
import rospy
import csv
import numpy as np
import os
from std_msgs.msg import ColorRGBA, Bool
from geometry_msgs.msg import PoseStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import tqdm
from hdmap_msgs.msg import HDmap

from hdmap_loader import hdmap_loader

class hdmap_localizer:
    def __init__(self, map_dir, link_type, surf_type, center, candid_num=100, thresh=2.0):
        self.map_dir = map_dir
        self.link_type = link_type
        self.surf_type = surf_type
        self.candid_num = candid_num
        self.thresh = thresh

        self.link_map = hdmap_loader(map_dir, link_type, center)
        self.surf_map = hdmap_loader(map_dir, surf_type, center)

        self.link_id = [i.record.ID for i in self.link_map.shaperecords]
        self.r_link_id = [i.record.R_LinkID for i in self.link_map.shaperecords]
        self.l_link_id = [i.record.L_LinKID for i in self.link_map.shaperecords]
        self.from_node_id = [i.record.FromNodeID for i in self.link_map.shaperecords]
        self.to_node_id = [i.record.ToNodeID for i in self.link_map.shaperecords]

        self.r_surf_id = [i.record.R_linkID for i in self.surf_map.shaperecords]
        self.l_surf_id = [i.record.L_linkID for i in self.surf_map.shaperecords]

        self.stop_shaperecords = [i for i in self.surf_map.shaperecords if i.record.Kind == '530']
        self.link_stop = self.link_stop_lane(0.2)
        
        self.center_shaperecords = [i for i in self.surf_map.shaperecords if i.record.Kind == '501']
        self.border_shaperecords = [i for i in self.surf_map.shaperecords if i.record.Kind == '505']

        self.centers = [i.shape.center for i in self.link_map.shaperecords]
        self.prev_link = None
        self.reliable = False

        self.init = False

        self.pub_vis = rospy.Publisher('/current_link_vis', MarkerArray, queue_size=10)
        self.pub_hdmap = rospy.Publisher('/hdmap/lane_info', HDmap, queue_size=10)
    
    def get_link(self, point):
        # coarse candidates
        d = (np.array(self.centers) - point) ** 2
        d = np.sqrt(d.sum(axis=1))
        c_candid_idx = np.argsort(d)[:self.candid_num]

        # fine candidates
        md = []
        for idx in c_candid_idx:
            d = (np.array(self.link_map.shaperecords[idx].shape.points) - point) ** 2
            d = np.sqrt(d.sum(axis=1))
            md.append(d.min())
        f_candid_idx = np.where(np.array(md) < self.thresh)[0]
        f_candid_idx = c_candid_idx[f_candid_idx]

        # sort candidates
        if len(f_candid_idx) == 1:
            self.reliable = True
            return f_candid_idx[0]
        else:
            if self.prev_link not in f_candid_idx:
                self.reliable = False
                if len(f_candid_idx) > 0:
                    self.prev_link = f_candid_idx.min()
                else:
                    self.prev_link = None
            return self.prev_link
    
    def link_stop_lane(self, thresh, r=2):
        rospy.loginfo('LINKING STOP LANES')

        # initialize
        end_link_points = np.array([i.shape.points[-1] for i in self.link_map.shaperecords])
        stop_points = [i.shape.points for i in self.stop_shaperecords]
        link_stop = [None] * len(self.link_map.shaperecords)
        for i, stop in enumerate(stop_points):
            stop = np.stack([stop] * len(end_link_points), axis=1)
            d = np.linalg.norm(stop - end_link_points, axis=-1).min(0)
            links = np.where(d < thresh)[0].tolist()
            for link in links:
                if link_stop[link] is None:
                    link_stop[link] = i
                else:
                    rospy.logwarn('Duplicate stop lane detected')
        
        # recursive
        for rec in range(r):
            for i, shaperecord in enumerate(self.link_map.shaperecords):
                if link_stop[i] is not None:
                    continue
                else:
                    next_idxs = np.where(np.array(self.from_node_id) == self.to_node_id[i])[0]
                    
                    stop_candid = list()
                    for next_idx in next_idxs:
                        if link_stop[next_idx] is not None:
                            stop_candid.append(link_stop[next_idx])

                        stop_idxs = np.unique(stop_candid)
                        if len(stop_idxs) == 1:
                            link_stop[i] = stop_idxs[0]
        return link_stop

    def link_description(self, current_link, id=0, use_reliable=None, color=None):
        if use_reliable is None and color is None:
            color = (0.5, 0.5, 0.5, 0.5)
        elif use_reliable is not None:
            color = (0.0, 0.0, 1.0, 1.0) if self.reliable is True else (1.0, 0.0, 0.0, 1.0)
        elif use_reliable is not None and color is not None:
            raise AttributeError('In function \'link_description\', use between use_reliable or color')
        shaperecord = self.link_map.shaperecords[current_link]
        marker = self.link_map.make_marker(shaperecord, id, color)
        return marker

    def pose_callback(self, msg):
        if self.init is False:
            self.link_map.hdmap_vis()
            self.surf_map.hdmap_vis()
            self.init = True
        markerarray = []
        lane_info = HDmap()
        lane_info.header.stamp = rospy.Time.now()
        lane_info.header.frame_id = 'map'
        lane_info.reliable = Bool(data=self.reliable)

        point = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        current_link = self.get_link(point)
    
        id = 0
        if current_link is not None:
            current_link_id = self.link_id[current_link]

            # current link
            current_lane_marker = self.link_description(current_link, id, use_reliable=True)
            lane_info.current_link
            markerarray.append(current_lane_marker)
            id += 1

            # left / right link
            left_link = self.link_id.index(self.l_link_id[current_link]) if self.l_link_id[current_link] != '' else None
            right_link = self.link_id.index(self.r_link_id[current_link]) if self.r_link_id[current_link] != '' else None

            if left_link is not None:
                left_link_marker = self.link_description(left_link, id, color=(0.0, 0.6, 1.0, 1.0))
                lane_info.left_link = left_link_marker
                markerarray.append(left_link_marker)

                id += 1
            if right_link is not None:
                right_link_marker = self.link_description(right_link, id, color=(1.0, 0.6, 0.0, 1.0))
                lane_info.right_link = right_link_marker
                markerarray.append(right_link_marker)
                id += 1

            # stop line
            current_stop_line = self.link_stop[current_link]
            if current_stop_line is not None:
                stop_shaperecord = self.stop_shaperecords[current_stop_line]
                points = stop_shaperecord.shape.points
                dist = np.linalg.norm(np.array(points) - point, axis=1).min()
                lane_info.stop_dist = dist

                stop_line_marker = self.surf_map.make_marker(stop_shaperecord, id, (0.5, 0.5, 1.0, 1.0))
                lane_info.stop_line = stop_line_marker
                markerarray.append(stop_line_marker)
                id += 1

            # left / right surf
            if current_link_id in self.l_surf_id:
                left_surf = self.l_surf_id.index(current_link_id)
                left_surf_marker = self.surf_map.make_marker(self.surf_map.shaperecords[left_surf], id, (0.5, 0.0, 0.5, 0.5))
                lane_info.left_line = left_surf_marker
                markerarray.append(left_surf_marker)
                id += 1

            if current_link_id in self.r_surf_id:
                right_surf = self.r_surf_id.index(current_link_id)
                right_surf_marker = self.surf_map.make_marker(self.surf_map.shaperecords[right_surf], id, (0.5, 0.5, 0.0, 0.5))
                lane_info.right_line = right_surf_marker
                markerarray.append(right_surf_marker)
                id += 1

            self.pub_vis.publish(MarkerArray(markerarray))
            self.pub_hdmap.publish(lane_info)
        else:
            rospy.logwarn('CURRENT LANE NOT FOUND')

if __name__ == '__main__':
    rospy.init_node('hdmap_localizer')
    pose_topic = rospy.get_param('~pose_topic')
    center_latitude = rospy.get_param('~center_latitude')
    center_longitude = rospy.get_param('~center_longitude')
    center_altitude = rospy.get_param('~center_altitude')
    localizer = hdmap_localizer(os.path.join(os.environ['AUTOWARE_DIR'], 'src', 'Data', 'Maps', 'HDmap'), 'link', 'surfacelinemark', (center_latitude, center_longitude, center_altitude))
    sub = rospy.Subscriber(pose_topic, PoseStamped, localizer.pose_callback)
    rospy.spin()


