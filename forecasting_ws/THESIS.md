# Robot simulation environment

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


## Running TSPTW planning from ZED cam

Start the ROS core
```
roscore
```

Run the ZED ros wrapper code. Note that that the parameters are set in `common.yaml` and `pub_frame_rate` is set to `60`.
```
roslaunch zed_wrapper zed2i.launch
```

Run the resizer node that rescales the image to the required format.
```
rosrun forecasting_tsptw image_resizer_node.py
```

Run the inference node that predicts hand positions by creating video tempfiles. 
```
rosrun forecasting_tsptw forecasting_gather_publisher.py
```

Run the TSPTW publisher node that generates robot task plan based on hand predictions.
```
rosrun forecasting_tsptw tsptw_publisher.py
```

Visualization nodes for TSPTW and hand predictions.
```
rosrun forecasting_tsptw publsih_tsptw_vis.py
```