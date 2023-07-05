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

from math import pi, tau, dist, fabs, cos

"""CONSTANTS:"""

# offset scale to move targets in orthogonal directions
DIST_SCALE = 0.02

HOME_JOINT_POSE = [-1.5664790887701328, -1.4255304565125417, -2.5255187899809455, -2.5519784422942475, -1.6016935564692556, 3.150484917203115]
POSE_1 = [-2.200340218567459, -2.0805938406156006, -1.7427202845207086, -2.734451289766482, -2.217025606956147, 2.989070354433986]
POSE_2 = [-1.728227373795077, -1.7867145550830044, -2.1741083746067913, -2.544794924096707, -1.7588379407638461, 3.114960675531224]
POSE_3 = [-0.9402553596452359, -1.7300824791315872, -2.2452055698186575, -2.569457997910006, -0.9916214780682919, 3.3026884231409217]
POSE_4 = [-0.5067101763963846, -2.0178442624052915, -1.8387706311145848, -2.857213982806627, -0.5443066560290424, 3.523212810474581]

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

    def go_to_joint_state(self, joint_pose):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = joint_pose[0]
        joint_goal[1] = joint_pose[1]
        joint_goal[2] = joint_pose[2]
        joint_goal[3] = joint_pose[3]
        joint_goal[4] = joint_pose[4]
        joint_goal[5] = joint_pose[5]

        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

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

            self.go_to_joint_state(HOME_JOINT_POSE)
            self.go_to_joint_state(POSE_1)
            self.go_to_joint_state(POSE_2)
            self.go_to_joint_state(POSE_3)
            self.go_to_joint_state(POSE_4)
            self.go_to_joint_state(HOME_JOINT_POSE)

            # input("Press `Enter` to begin the checkbot by setting up the moveit_commander ...")
            # print("----------------------------------------------------------")
            # input("Press `Enter` to go to a home pose goal ...")
            # self.go_to_joint_state(HOME_JOINT_POSE)
            # self.go_to_pose(HOME_POSE)

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