#!/usr/bin/env python3

# Import python libs
import os
import cv2
from datetime import datetime

# import ROS libraries
import rospy
import rospkg
from cv_bridge import CvBridge
from moveit_commander import MoveItCommanderException

# Import pre-defined msg/srv/actions
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image

# Custom classes
from bot_movegroup import BotMovegroup

# Constants/tunable parameters
PKG_NAME = "bot_interface"
ABS_PATH = f"{rospkg.RosPack().get_path(PKG_NAME)[: -len(PKG_NAME)-4]}resources/"
CAMERA_SERVICE = "/camera/color/image_raw"

## Given data - Target joint goals to execute.
# execute_traj will have to iterate through this
JOINT_STATES = {
    "POS_HOME": [-90, -60, -78, -78, 91, 0],
    "POS_ONE": [-36, -173, -9, -103, 9, 9],
    "POS_TWO": [-122, -173, -20, -7, 143, 9],
    "__POS_HOME": [-90, -60, -78, -78, 91, 0],
}


class BotInterface(BotMovegroup):
    def __init__(self):
        super().__init__()
        rospy.init_node("bot_interface", anonymous=True)

        self.bridge = CvBridge()
        self.srv_picture = rospy.Service(
            "bot_interface/click_picture", Trigger, self.trigger_response_click_picture
        )
        self.srv_trajectory = rospy.Service(
            "bot_interface/execute_traj", Trigger, self.trigger_response_execute_traj
        )

        rospy.loginfo("Initialized BotInterface")

    def trigger_response_click_picture(self, request):
        """
        Click picture service callback.
        This service will take a picture and save it to the resources folder.
        """
        resp = TriggerResponse()

        try:
            img_msg = rospy.wait_for_message(CAMERA_SERVICE, Image, 5)
        except rospy.ROSException as e:
            resp.success = False
            resp.message = f"Service {CAMERA_SERVICE} not available!"
            return resp

        # try to convert img to cv
        try:
            img_cv = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        except cv2.CvBridgeError as e:
            resp.success = False
            resp.message = "Camera image could not be converted!"
            return resp

        # generate img name
        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        img_name = f"camera_image_{date_time}.png"
        img_path = ABS_PATH + img_name

        if not os.path.exists(ABS_PATH):
            os.makedirs(ABS_PATH)

        cv2.imwrite(img_path, img_cv)

        resp.success = True
        resp.message = f"Succesfully saved image {img_name}!"

        return resp

    def trigger_response_execute_traj(self, request):
        """
        Execute trajectory service callback.
        This service will take a trajectory and execute it.
        """
        resp = TriggerResponse()

        try:
            for pos in JOINT_STATES:
                self.execute_traj(JOINT_STATES[pos])

        except MoveItCommanderException as e:
            resp.message = "Could not execute trajectory!"
            resp.success = False
            return resp

        resp.message = "Succesfully executed trajectory!"
        resp.success = True
        return resp


def main():
    mcc = BotInterface()
    mcc.interface_primary_movegroup()
    rospy.spin()


if __name__ == "__main__":
    main()
