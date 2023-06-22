#!/usr/bin/env python3

# Std python libs
import sys
from math import radians

# import ROS libraries
import rospy
import moveit_commander

# Failsafe
rospy.wait_for_service("/move_group/trajectory_execution/set_parameters")

class BotMovegroup:

    def __init__(self):
        # initialize moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

    def interface_primary_movegroup(self):
        """
        Interfaces with the move_group instance with the help of the
        move_group commander
        """
        # Interface with move group 
        self.robot , self.planning_scene, self.move_group = self._initialize_planner()

    def _initialize_planner(self):

        # instantiate the robot commander
        robot = moveit_commander.RobotCommander()

        # instantiate the scene. Needed to get, set and update 
        # the robots surrounding of the outside world
        scene = moveit_commander.PlanningSceneInterface()

        # Instantiate the move group object - this is the one 
        # that takes care of planning and execution
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # Initialize planning parameters
        move_group.allow_replanning(True)

        return robot, scene, move_group

    def execute_traj(self, target_joints):
        """
        Given a target joint state, plans and executes the trajectory
        with moveit
        """
        self.move_group.set_start_state_to_current_state()
        target_joints_rad = BotMovegroup._convert_to_radians(target_joints)
        self.move_group.set_joint_value_target(target_joints_rad)
        return self.move_group.go()

    @classmethod
    def _convert_to_radians(cls, target_joints):
        return [radians(i) for i in target_joints]