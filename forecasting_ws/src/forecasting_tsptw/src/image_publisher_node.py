#!/usr/bin/env python3

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    rospy.init_node('image_publisher_node', anonymous=True)
    pub = rospy.Publisher('image_topic', Image, queue_size=10)
    bridge = CvBridge()
    rospy.loginfo("Image publisher node started")
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 340)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 340)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rospy.loginfo("Frame width: {}, frame height: {}".format(width, height))

    new_height = 340
    new_width = int(width * (new_height / height))
    dim = (new_width, new_height)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            res_frame = cv2.resize(frame, dim)
            image_message = bridge.cv2_to_imgmsg(res_frame, encoding='bgr8')
            pub.publish(image_message)
        rate.sleep()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
