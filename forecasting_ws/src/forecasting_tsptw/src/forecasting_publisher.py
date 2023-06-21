#!/usr/bin/env python3

import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image


class ForecastingPublisher:
    def __init__(self):
        rospy.init_node('forecasting_publisher_node', anonymous=True)
        self.pub = rospy.Publisher(
            'forecasting_topic', numpy_msg(Floats), queue_size=1)
        rospy.Subscriber('image_topic', Image, self.image_callback)
        rospy.loginfo("Forecasting publisher node started")

    def image_callback(self, image_message):
        msg = np.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5], dtype=np.float32)
        self.pub.publish(msg)


if __name__ == '__main__':
    try:
        forecasting_publisher = ForecastingPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
