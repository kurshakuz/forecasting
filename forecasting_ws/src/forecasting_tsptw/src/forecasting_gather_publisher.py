#!/usr/bin/env python3

import numpy as np
import rospy
import os
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
import sys

worksapce_path = '/home/dev/workspace/thesis-ws'

sys.path.append(os.path.join(worksapce_path))
from video_sampler import create_tempfile
from inference import Inferencer

sys.path.append(os.path.join(worksapce_path, 'uniformer'))
import volume_transforms as volume_transforms
import video_transforms as video_transforms
import modeling_finetune_uniformer_ego4d

sys.path.append(os.path.join(worksapce_path, 'tsptw'))
from main_hands import hands_tsptw_solver

class ForecastingPublisher:
    def __init__(self, snippet_length, overlap_length):
        self.initialized = False
        rospy.init_node('forecasting_publisher_node', anonymous=True)
        self.pub = rospy.Publisher('forecasting_topic', Float64MultiArray, queue_size=1)
        self.image_sub = rospy.Subscriber('image_topic', Image, self.image_callback)
        rospy.loginfo("Forecasting publisher node started")

        self.fps = 30  # Assuming the webcam captures at 30 frames per second
        self.snippet_frames = snippet_length * self.fps
        self.overlap_frames = overlap_length * self.fps

        self.buffer_frames = []
        self.snippet_num = 1
        self.frame_num = 0

        print("Loading model...")
        self.inferencer = Inferencer()
        self.initialized = True

    def image_callback(self, image_message):
        if not self.initialized:
            return

        try:
            frame = np.frombuffer(image_message.data, dtype=np.uint8).reshape(image_message.height, image_message.width, -1)
        except ValueError:
            rospy.logwarn("Failed to convert image message to numpy array")
            return

        self.frame_num += 1
        self.buffer_frames.append(frame)

        if len(self.buffer_frames) > self.snippet_frames:
            self.buffer_frames.pop(0)

        if len(self.buffer_frames) == self.snippet_frames:
            if self.frame_num % self.overlap_frames == 0:
                snippet = self.buffer_frames.copy()
                temp_file_path = create_tempfile(snippet)
                result = self.inferencer.predict_in_memory(temp_file_path)[0]
                print(result)
                self.snippet_num += 1

                data_list = result.tolist()
                float_array = Float64MultiArray()
                float_array.layout.dim = []
                float_array.layout.data_offset = 0
                float_array.data = data_list

                self.pub.publish(float_array)


if __name__ == '__main__':
    snippet_length = 2  # in seconds
    overlap_length = 1  # in seconds

    try:
        forecasting_publisher = ForecastingPublisher(snippet_length, overlap_length)
        rospy.spin()
    except rospy.ROSInterruptException:
        if forecasting_publisher.tempfile is not None:
            forecasting_publisher.tempfile.close()
            os.unlink(forecasting_publisher.tempfile.name)
