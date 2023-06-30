#include "ros/ros.h"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>

// A quick and dirty interface to move the robot

int main(int argc, char **argv)
{
	ros::init(argc, argv, "move_group_interface");
	ros::NodeHandle node_handle;

	ros::AsyncSpinner spinner(1);
	spinner.start();

	static const std::string PLANNING_GROUP = "manipulator";
	moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);

	moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

	const moveit::core::JointModelGroup *joint_model_group =
		move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

	move_group_interface.setMaxVelocityScalingFactor(0.1);
	move_group_interface.setMaxAccelerationScalingFactor(0.1);

	moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState();
	std::vector<double> joint_group_up = { 0, -1.5708, 0,-1.5708, 0, 0};
	std::vector<double> joint_group_one = { -1.90, -0.349, -2.059,-1.675,1.535, 0};
	std::vector<double> joint_group_two = { -1.117, -0.349, -2.059,-1.675,1.535, 0};
	std::vector<double> joint_group_three = { -0.157, -1.780, -2.042,-1.6755,0.279, 0.157};
	std::vector<double> joint_group_four = { -0.7504, -1.309, -1.4835,-1.2566,0.820, 0.157};
  	move_group_interface.setJointValueTarget(joint_group_up);
	move_group_interface.move();
  	// move_group_interface.setJointValueTarget(joint_group_one);
	// move_group_interface.move();
  	// move_group_interface.setJointValueTarget(joint_group_two);
	// move_group_interface.move();
  	// move_group_interface.setJointValueTarget(joint_group_three);
	// move_group_interface.move();
  	// move_group_interface.setJointValueTarget(joint_group_four);
	// move_group_interface.move();
	ros::shutdown();
	return 0;
}