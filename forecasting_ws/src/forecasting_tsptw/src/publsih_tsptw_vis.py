#!/usr/bin/env python3

import os
import sys
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

worksapce_path = '/home/shyngys/workspace/thesis-ws'
sys.path.append(os.path.join(worksapce_path, 'tsptw'))
from hands_helpers import define_regions_and_intersections


def draw_rectangles(img, rectangles, color):
    for rectangle in rectangles:
        x, y, width, height = int(rectangle.x), int(rectangle.y), int(rectangle.width), int(rectangle.height)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 3)

def draw_hands(rectangles, squares, intersecting):
    img = np.zeros([320, 320 * 5, 3], dtype=np.uint8)
    img.fill(255)

    draw_rectangles(img, rectangles, (0, 255, 0))  # Green color for rectangles

    # Line separators
    for i in range(4):
        cv2.line(img, (320 * (i + 1), 0), (320 * (i + 1), 320), (0, 0, 0), 3)  # Black color for lines

    draw_rectangles(img, squares, (255, 0, 0))  # Blue color for hands

    # Plotting the intersecting rectangles
    draw_rectangles(img, intersecting, (0, 0, 255))  # Red color for intersecting rectangles

    return img

def draw_hands_on_image(img, rectangles, squares, intersecting):
    draw_rectangles(img, rectangles, (0, 255, 0))  # Green color for rectangles

    # draw_rectangles(img, squares, (255, 0, 0))  # Blue color for hands

    # Plotting the intersecting rectangles
    draw_rectangles(img, intersecting, (0, 0, 255))  # Red color for intersecting rectangles

    return img

class TSPTWVisPubNode:
    def __init__(self):
        rospy.init_node('tsptw_vis_publisher_node')

        self.forecasting_topic = '/forecasting_topic'
        self.subscriber_forecasting = rospy.Subscriber(self.forecasting_topic, Float64MultiArray, self.callback_forecasting)
        self.tsptw_topic = '/tsptw_topic'
        self.subscriber_tsptw = rospy.Subscriber(self.tsptw_topic, Float64MultiArray, self.callback_tsptw)
        self.image_sub = rospy.Subscriber('/zed2i_resized_raw', Image, self.image_callback)

        self.vis_publisher = rospy.Publisher('tsptw_vis', Image, queue_size=10)

        self.overlay_vis_publisher = rospy.Publisher('overlay_tsptw_vis', Image, queue_size=10)
        self.cv_bridge = CvBridge()

        self.frame_width = 320
        self.num_regions = 4

        self.last_preds = None

    def callback_forecasting(self, msg):
        preds = np.array(msg.data).reshape(1, -1)
        # print(preds)

        reg_preds = preds[:,:20]
        contact_pred = preds[:,20]
        reg_preds_pairs = np.split(reg_preds, 5, axis=1)
        rectangles, squares, intersecting = define_regions_and_intersections(self.frame_width, self.num_regions, reg_preds_pairs)
        img = draw_hands(rectangles, squares, intersecting)
        formatted_msg = self.cv_bridge.cv2_to_imgmsg(img, 'bgr8')
        self.vis_publisher.publish(formatted_msg)

        self.last_rectangles, self.last_squares, self.last_intersecting = rectangles, squares, intersecting
        self.last_preds = preds

    def callback_tsptw(self, msg):
        plan = np.array(msg.data).reshape(5, 5)
        print(plan)
        self.last_plan = plan

    def image_callback(self, msg):
        if self.last_preds is None or self.last_rectangles is None:
            return

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr('Error converting Image message: {}'.format(e))
            return

        img = draw_hands_on_image(cv_image, self.last_rectangles, self.last_squares, self.last_intersecting)
        formatted_msg = self.cv_bridge.cv2_to_imgmsg(img, 'bgr8')
        self.overlay_vis_publisher.publish(formatted_msg)

if __name__ == '__main__':
    try:
        node = TSPTWVisPubNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
