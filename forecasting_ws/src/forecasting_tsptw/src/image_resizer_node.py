#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageResizerNode:
    def __init__(self):
        rospy.init_node('image_resizer_node')
        print("Resizer node started")
        self.image_sub = rospy.Subscriber('/zed2i/zed_node/left_raw/image_raw_color', Image, self.image_callback)
        self.image_pub = rospy.Publisher('zed2i_resized_raw', Image, queue_size=10)
        self.cv_bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr('Error converting Image message: {}'.format(e))
            return

        resized_image = self.resize_image(cv_image, 320)
        cropped_image = self.center_crop(resized_image)

        resized_msg = self.cv_bridge.cv2_to_imgmsg(cropped_image, 'bgr8')
        self.image_pub.publish(resized_msg)

    def resize_image(self, image, size):
        height, width = image.shape[:2]
        if height < width:
            new_height = size
            new_width = int(float(width) / height * size)
        else:
            new_width = size
            new_height = int(float(height) / width * size)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    def center_crop(self, image):
        height, width = image.shape[:2]
        crop_size = min(height, width)
        y = (height - crop_size) // 2
        x = (width - crop_size) // 2
        cropped_image = image[y:y+crop_size, x:x+crop_size]
        return cropped_image

if __name__ == '__main__':
    image_resizer = ImageResizerNode()
    rospy.spin()
