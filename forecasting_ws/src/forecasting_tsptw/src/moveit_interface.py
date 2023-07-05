#!/usr/bin/env python3

import sys
import time

import rospy

import geometry_msgs.msg
import moveit_commander
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PoseStamped, WrenchStamped, Pose

from six.moves import input

from moveit_commander.exception import MoveItCommanderException

from all_close import all_close

"""CONSTANTS:"""

# offset scale to move targets in orthogonal directions
DIST_SCALE = 0.02

HOME_POSE = [0.18059420288171502, 0.3219906149681194, 0.30261826442649115, -0.7692760100441558, -0.008125941611489511, -0.006503281370102989, 0.638831821980474]

class CheckBotTracker:
    """CheckBotTracker"""

    def __init__(self, dist_scale: float):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("checkbot_tracker", anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        curr_pose = self.move_group.get_current_pose().pose

        print("Planning frame: %s" % self.planning_frame)
        print("End effector link: %s" % self.eef_link)
        print("Available Planning Groups: {0}".format(self.robot.get_group_names()))
        print(self.robot.get_current_state())
        print("Printing robot state \n")
        print(
            "Printing robot pose: [{0}, {1}, {2}, {3}, {4}, {5}, {6}] \n".format(
                curr_pose.position.x,
                curr_pose.position.y,
                curr_pose.position.z,
                curr_pose.orientation.x,
                curr_pose.orientation.y,
                curr_pose.orientation.z,
                curr_pose.orientation.w,
            )
        )

        # Create tf2 transform listener
        # print("============ Setting up transform listener")
        # self.tf_buffer = tf2_ros.Buffer()
        # listener = tf2_ros.TransformListener(self.tf_buffer)
        # print("")

        # self.tfd_poses_pub = rospy.Publisher(
        #     "tfd_tracked_pose", PoseStamped, queue_size=10
        # )

        # # Subscribe to tracker topic
        # print("Subscribing to the tracker topic \n")
        # rospy.Subscriber(
        #     "/track_target",
        #     PoseStamped,
        #     self.track_target_callback,
        # )
    
    def track_target_callback(self, msg):
        from_frame = "tracked_obj"
        to_frame = "base_link"
        self.transformed_pose = self.transform_pose(msg, from_frame, to_frame)

    def transform_pose(self, input_pose, from_frame, to_frame):
        pose_stamped = input_pose

        try:
            output_pose_stamped = self.tf_buffer.transform(
                pose_stamped, to_frame, rospy.Duration(1)
            )
            self.tfd_poses_pub.publish(output_pose_stamped)
            return output_pose_stamped.pose
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

    def go_to_pose(self, input_pose):
        if type(input_pose) == list:
            pose = geometry_msgs.msg.Pose()
            pose.position.x = input_pose[0]
            pose.position.y = input_pose[1]
            pose.position.z = input_pose[2]

            pose.orientation.x = input_pose[3]
            pose.orientation.y = input_pose[4]
            pose.orientation.z = input_pose[5]
            pose.orientation.w = input_pose[6]
        else:
            pose = geometry_msgs.msg.Pose()
            pose.position = input_pose.position
            curr_pose = self.move_group.get_current_pose().pose
            pose.orientation = curr_pose.orientation

        self.move_group.set_pose_target(pose)

        # Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()

        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()

        # if ~success:
        #     raise MoveItCommanderException
        # workaround to avoid damage TODO: does not work for already reached goal

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose, current_pose, 0.01)

    def run(self):
        try:
            print("\n----------------------------------------------------------")
            print("Welcome to the Pushbot Interface")
            print("----------------------------------------------------------")
            print("Press Ctrl-D to exit at any time\n")

            # input("Press `Enter` to begin the checkbot by setting up the moveit_commander ...")
            # print("----------------------------------------------------------")
            # input("Press `Enter` to go to a home pose goal ...")
            self.go_to_pose(HOME_POSE)

            # input("Press `Enter` to go to a lower goal ...")
            # self.go_to_pose(TARGET_POSE)

            # input("Press `Enter` to go to a tracker goal ...")
            # # self.go_to_pose(self.transformed_pose)
            # self.go_to_pose(NEW_TARGET_POSE)

            # rate = rospy.Rate(1)
            # input("Press `Enter` to go to a tracker goal ...")
            # for i in range(50):
            #     print("Going to pose ")
            #     print(self.transformed_pose)
            #     self.go_to_pose(self.transformed_pose)
            #     rate.sleep()

            # # input("Press `Enter` to go to a home pose goal ...")
            # self.go_to_pose(HOME_POSE)

            print("Tracking execution complete!")

        except rospy.ROSInterruptException:
            return
        except KeyboardInterrupt:
            return
        except MoveItCommanderException:
            return


if __name__ == "__main__":
    checkbot_tracker = CheckBotTracker(DIST_SCALE)
    checkbot_tracker.run()