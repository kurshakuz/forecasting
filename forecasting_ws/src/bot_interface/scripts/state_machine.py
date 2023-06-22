#!/usr/bin/env python3

import os
from time import sleep

import rospy
import smach_ros
from smach import State, StateMachine
from std_srvs.srv import Trigger

ABS_PATH = '< absolute path to your catkin workspace >/resources'


class Resting(State):
    def __init__(self):
        State.__init__(self, outcomes=[], input_keys=['input'])

    def execute(self, userdata):
        sleep(1)

        rospy.loginfo('Executing state RESTING')

        enter = input("Enter '1' to start execution: ")
        if enter == '1':
            raise NotImplementedError
        else:
            raise NotImplementedError


class ServiceVerification(State):
    """
    Make sure the robot is booted via 'ur_gazebo ur10e_bringup.launch'
    and ready to receive commands via 'ur10e_moveit_config ur10e_moveit_planning_execution.launch'
    """

    def __init__(self):
        State.__init__(self, outcomes=[], input_keys=['input'])

    def execute(self, userdata):
        sleep(1)

        rospy.loginfo('Executing state SERVICE_VERIFICATION')

        raise NotImplementedError


class Moving(State):
    def __init__(self):
        State.__init__(self, outcomes=[], input_keys=['input'])

    def execute(self, userdata):
        sleep(1)

        rospy.loginfo('Executing state MOVING')

        raise NotImplementedError


class ClickingPicture(State):
    def __init__(self):
        State.__init__(self, outcomes=[], input_keys=['input'])

    def execute(self, userdata):
        sleep(1)

        rospy.loginfo('Executing state CLICKING_PICTURE')

        raise NotImplementedError


def main():
    rospy.init_node('robot_state_machine')
    sm = StateMachine(outcomes=['succeeded', 'failed'])
    sm.userdata.sm_input = 0

    with sm:
        StateMachine.add('RESTING', Resting(), transitions={}, remapping={'input': 'sm_input'})
        StateMachine.add('SERVICE_VERIFICATION', ServiceVerification(), transitions={}, remapping={'input': 'sm_input'})
        StateMachine.add('MOVING', Moving(), transitions={}, remapping={'input': 'sm_input'})
        StateMachine.add('CLICKING_PICTURE', ClickingPicture(), transitions={}, remapping={'input': 'sm_input'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()

    rospy.spin()
    sis.stop()


if __name__ == '__main__':
    main()
