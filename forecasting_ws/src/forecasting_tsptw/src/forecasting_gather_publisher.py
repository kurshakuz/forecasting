#!/usr/bin/env python3

import numpy as np
import rospy
import cv2
import os
import tempfile
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image


def create_tempfile(snippet_frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = snippet_frames[0].shape
    new_height = 340
    new_width = int(width * (new_height / height))
    dim = (new_width, new_height)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        output_name = temp_file.name
        out = cv2.VideoWriter(output_name, fourcc, 30.0, dim)

        for frame in snippet_frames:
            res_frame = cv2.resize(frame, dim)
            out.write(res_frame)

        out.release()
        print(f'Snippet {output_name} created.')

    return output_name



class ForecastingPublisher:
    def __init__(self, snippet_length, overlap_length):
        rospy.init_node('forecasting_publisher_node', anonymous=True)
        self.pub = rospy.Publisher('forecasting_topic', numpy_msg(Floats), queue_size=1)
        self.image_sub = rospy.Subscriber('image_topic', Image, self.image_callback)
        rospy.loginfo("Forecasting publisher node started")


        self.fps = 30  # Assuming the webcam captures at 30 frames per second
        self.snippet_frames = snippet_length * self.fps
        self.overlap_frames = overlap_length * self.fps

        self.buffer_frames = []
        self.snippet_num = 1
        self.frame_num = 0


    def image_callback(self, image_message):
        try:
            frame = np.frombuffer(image_message.data, dtype=np.uint8).reshape(image_message.height, image_message.width, -1)
        except ValueError:
            rospy.logwarn("Failed to convert image message to numpy array")
            return
        
        # rospy.loginfo("Received image message with shape {}".format(frame.shape))

        self.frame_num += 1
        self.buffer_frames.append(frame)

        if len(self.buffer_frames) > self.snippet_frames:
            self.buffer_frames.pop(0)

        if len(self.buffer_frames) == self.snippet_frames:
            if self.frame_num % self.overlap_frames == 0:
                snippet = self.buffer_frames.copy()
                temp_file_path = create_tempfile(snippet)
                self.snippet_num += 1

        # Publish a sample message for demonstration
        msg = np.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5], dtype=np.float32)
        self.pub.publish(msg)


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
