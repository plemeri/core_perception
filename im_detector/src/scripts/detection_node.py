#!/usr/bin/python3
import time

import easyocr
import numpy as np
import os
import cv2

import torch
from torchvision.transforms.functional import to_pil_image

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
# from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from autoware_msgs.msg import DetectedObjectArray, DetectedObject

from detect import *


IMAGE_SIZE = (960, 1280)
DEBUG = False
TS_LIST = ['10', '15', '20', '25', '30', '35', '40', '45', '50']
weights = os.path.join(os.environ['AUTOWARE_DIR'], 'src/Data/yolo/best_yolov4-csp-coco-tune.pt')
cfg = os.path.join(os.environ['AUTOWARE_DIR'], 'src/Data/yolo/yolov4-csp-coco-tune.cfg')
tunes = os.path.join(os.environ['AUTOWARE_DIR'], 'src/Data/yolo/Tune.names')

def init():
    global colors
    global device
    global detector
    global digit_reader
    global names

    device = torch.device(0)

    # Load models
    detector = Darknet(cfg, 640).cuda()
    try:
        detector.load_state_dict(torch.load(weights, map_location=device)['model'])
    except:
        detector = detector.to(device)
        load_darknet_weights(detector, weights)
    detector.to(device).eval()

    detector.half()  # to FP16
    digit_reader = easyocr.Reader(['en'], gpu=True)

    # Get names and colors
    with open(tunes, 'r') as f:
        names = f.read().split('\n')
    names = list(filter(None, names))
    names[-1] = "Red"
    names.append("Yellow")
    names.append("Green")
    colors = [[np.random.randint(0, 256) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, 960, 1280), device=device)  # init img
    _ = detector(img.half()) if device.type != 'cpu' else None  # run once
      

@torch.no_grad()
def image_callback(msg):
    global colors
    global detector
    global digit_reader
    global names

    # eun0
    tl_detected = False
    ts_detected = False

    img = msg.data
    img = np.frombuffer(img, dtype=np.uint8)
    img = img.reshape((*IMAGE_SIZE, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1)
    img = img.half()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    preds = detector(img)[0]

    # Apply NMS
    preds = non_max_suppression(preds, 0.4, 0.5, classes = None, agnostic=False)[0]

    if DEBUG:
        viz = img.type(torch.float32).cpu()
        viz = to_pil_image(viz[0])
        viz = np.array(viz)
        # eun0
        ts_res = np.zeros((150, 150, 3), dtype=np.uint8)
        tl_img = np.zeros((50, 150, 3), dtype=np.uint8)
    
    if preds is not None:
        det2darr = DetectedObjectArray()
        det2darr.header = msg.header

        max_size = 0.0
        ts_data = "Empty"

        for pred in preds:
            pred = pred.cpu().numpy()
            det2d = DetectedObject()
            
            cls_id = int(pred[5])

            # Speed Limit
            if cls_id == 5 or cls_id == 6 or cls_id == 7:
                # eun0
                if pred[0] < 640 or pred[2] - pred[0] < 10:
                    continue
                ts_detected = True
                #
                # if pred[0] < 640:
                    # continue
                cpu_img = img[0].float().cpu()
                pil_img = to_pil_image(cpu_img)
                pil_img = pil_img.crop((pred[0], pred[1], pred[2], pred[3]))
                pil_img = pil_img.resize((150, 150))
                ts_img = np.asarray(pil_img)
                result = digit_reader.readtext(
                    ts_img,
                    allowlist=['0', '1', '2', '3', '4', '5'],
                    detail=0
                    )
                if len(result) == 0:
                    result.append("Empty")
                
                curr_size = (pred[2] - pred[0]) * (pred[3] - pred[1])
                if curr_size > max_size:
                    ts_data = result[0]
                    max_size = curr_size
                     # eun0
                    ts_res = np.array(pil_img)
                    cv2.putText(ts_res,
                                f"{ts_data} {pred[4]:.2f}", (0, 50),
                                0, 1, (255, 255, 255), 1, cv2.LINE_AA)
                
                
            # Traffic Light
            if cls_id >= 10:
                tf_width = pred[2] - pred[0]
                tf_height = pred[3] - pred[1]

                # skip false positive - ratio
                if(tf_width < tf_height) : 
                    if DEBUG: print('skip ratio : ', pred[0], pred[1], pred[2], pred[3])
                    continue
                # skip false positive - ROI
                if(pred[1] > 450) :
                    if DEBUG: print('skip ROI : ', pred[0], pred[1], pred[2], pred[3])
                    continue

                

                cpu_img = img[0].float().cpu()
                pil_img = to_pil_image(cpu_img)

                #crop image by bbox : 80%
                # pil_img = pil_img.crop((pred[0], pred[1], pred[2], pred[3]))
                width_crop_ratio = 0.1
                height_crop_ratio = 0.2
                pil_img = pil_img.crop((pred[0] + (width_crop_ratio*tf_width), pred[1] + (height_crop_ratio*tf_height), pred[2] - (width_crop_ratio*tf_width), pred[3] - (height_crop_ratio*tf_height)))
                pil_img = pil_img.resize((150, 50))
                tl_img = np.asarray(pil_img)
                gray = cv2.cvtColor(tl_img, cv2.COLOR_BGR2GRAY)
                red = tl_img[:, :50:, 2].mean()
                yellow = gray[:, 50:100].mean()
                green = tl_img[:, 100:, 1].mean()
                # print(f"{red:.2f}, {yellow:.2f}, {green:.2f}")

                # skip false positive - color
                if red < 50.0 and yellow < 50.0 and green < 50.0:
                    if DEBUG: print('skip color : ', red, yellow, green)
                    continue

                # skip false positive - red color
                red_center = tl_img[20:30, 20:30, 2].mean()
                yellow_center = gray[20:30, 70:80].mean()
                green_center = tl_img[20:30, 20:30, 1].mean()

                red_zone_green_center = tl_img[20:30, 20:30, 1].mean()
                red_zone_blue_center  = tl_img[20:30, 20:30, 0].mean()
                # if (red_zone_green_center - red_center) > 0 or (red_zone_blue_center - red_center) > 0 :
                if red_zone_green_center > 30 or red_zone_blue_center > 30 :
                    if DEBUG: print('skip red color : ', red_center, red_zone_green_center, red_zone_blue_center)
                    continue

                cls_id = 10 + np.argmax([red, yellow, green])

                # DEBUG
                if DEBUG:
                    cv2.putText(tl_img,
                                f"{names[cls_id]} {pred[4]:.2f}", (0, 50),
                                0, 1, (255, 255, 255), 1, cv2.LINE_AA)

                # Publish traffic light.
                traffic_light = String()
                traffic_light.data = names[cls_id]
                traffic_light_pub.publish(traffic_light)
                # eun0
                tl_detected = True

            # Class probabilities.
            det2d.id = cls_id
            det2d.score = pred[4]

            # The 2d data that generated these results.
            # det2d.source_img = msg

            # 2D bouding box surrounding the object.
            det2d.x = int((pred[0] + pred[2]) * .5)
            det2d.y = int((pred[1] + pred[3]) * .5)
            det2d.width = int(pred[2] - pred[0])
            det2d.height = int(pred[3] - pred[1])

            det2darr.objects.append(det2d)
            
            if DEBUG:
                xyxy = (pred[0], pred[1], pred[2], pred[3])
                label = '%s %.2f' % (names[det2d.id], det2d.score)  # names[int(cls)], conf
                plot_one_box(xyxy, viz, label=label, color=colors[cls_id], line_thickness=3)

        if len(det2darr.objects) >= 1:
            det_pub.publish(det2darr)

        if ts_data in TS_LIST:
            speed_limit = String()
            speed_limit.data = ts_data
            speed_limit_pub.publish(speed_limit)

    # eun0
    if not tl_detected:
        traffic_light = String()
        traffic_light.data = '0'
        traffic_light_pub.publish(traffic_light)
    if not ts_detected:
        speed_limit = String()
        speed_limit.data = '0'
        speed_limit_pub.publish(speed_limit)
    #
    
    if DEBUG:
        cv2.imshow("test", viz)
        cv2.imshow("traffic sign", ts_res)
        cv2.imshow("traffic light", tl_img)
        cv2.waitKey(1)


def info_callback(msg):
    IMAGE_SIZE = (msg.height, msg.width)


if __name__ == '__main__':
    init()
    rospy.init_node('im_detector')
    image_topic = rospy.get_param('~image_topic', '/color/image_raw')
    camera_info_topic = rospy.get_param('~camera_info_topic', '/color/camera_info')
    detection_topic = rospy.get_param('~detection_topic', '/im_detector/objects')
    
    image_sub = rospy.Subscriber(image_topic, ImageMsg, image_callback, tcp_nodelay=True)
    info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, info_callback)
    
    det_pub = rospy.Publisher(detection_topic, DetectedObjectArray, queue_size=10)
    speed_limit_pub = rospy.Publisher('/ts_state', String, queue_size=10)
    traffic_light_pub = rospy.Publisher('/tl_state', String, queue_size=10)
    rospy.spin()