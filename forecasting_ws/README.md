# Robot simulation environment for thesis

Run commands in the following order:
```
roslaunch ur_gazebo ur10e_bringup.launch
```

```
roslaunch ur10e_moveit_config ur10e_moveit_planning_execution.launch use_rviz:=true
```

```
roslaunch create_environment scan_map.launch
```
