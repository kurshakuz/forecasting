#!/usr/bin/env python3

import os
import sys
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray


worksapce_path = '/home/shyngys/workspace/thesis-ws'
sys.path.append(os.path.join(worksapce_path, 'tsptw'))
from main_hands import hands_tsptw_solver


class NumpyMsgRepublisherNode:
    def __init__(self):
        rospy.init_node('tsptw_publisher_node')

        # Subscribe to the input topic
        self.input_topic = '/forecasting_topic'
        self.subscriber = rospy.Subscriber(self.input_topic, Float64MultiArray, self.callback)

        # Publish on the output topic
        self.output_topic = '/tsptw_topic'
        self.publisher = rospy.Publisher(self.output_topic, Float64MultiArray, queue_size=1)

        self.iter_max = 30
        self.level_max = 8
        self.initial_path_type = 'rdy'

    def callback(self, msg):
        preds = np.array(msg.data).reshape(1, -1)
        plan = hands_tsptw_solver(self.iter_max, self.level_max, self.initial_path_type, preds=preds)
        print(plan)

        data_list = plan.flatten().tolist()
        float_array = Float64MultiArray()
        float_array.layout.dim = []
        float_array.layout.data_offset = 0
        float_array.data = data_list
        self.publisher.publish(float_array)

if __name__ == '__main__':
    try:
        node = NumpyMsgRepublisherNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
