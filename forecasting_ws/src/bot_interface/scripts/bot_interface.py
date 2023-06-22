#!/usr/bin/env python3

# Import python libs
import os

# import ROS libraries
import rospy

# Import pre-defined msg/srv/actions
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image

# Custom classes
from bot_movegroup import BotMovegroup

# Constants/tunable parameters
ABS_PATH = '< absolute path to your catkin workspace >/resources'

## Given data - Target joint goals to execute.
# execute_traj will have to iterate through this
JOINT_STATES = {
                'POS_HOME': [-90, -60, -78, -78, 91, 0],
                'POS_ONE': [-36, -173, -9, -103, 9, 9],
                'POS_TWO': [-122, -173, -20, -7, 143, 9],
                '__POS_HOME': [-90, -60, -78, -78, 91, 0],
                }

class BotInterface(BotMovegroup):
    def __init__(self):
        super().__init__()
        rospy.init_node('bot_interface', anonymous=True)

        ##############
        # Question 3 - Create 2 service servers and their callbacks here
        # Use the already implemented functions from the parent class for
        # trajectory execution
        # Also remember to define ABS_PATH
        raise NotImplementedError
        ##############

    def trigger_response_click_picture(self, request):
        """
        Click picture service callback.
        This service will take a picture and save it to the resources folder.
        """

        raise NotImplementedError
    
    def trigger_response_execute_traj(self, request):
        """
        Execute trajectory service callback.
        This service will take a trajectory and execute it.
        """

        raise NotImplementedError

def main():
    mcc = BotInterface() 
    mcc.interface_primary_movegroup()
    rospy.spin()
    
if __name__ == "__main__":
    main()