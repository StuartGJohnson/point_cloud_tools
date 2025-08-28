# point_cloud_tools

ROS2 c++ package for point cloud operations.

## Motivation

In bringing up an Intel Realsense depth camera in simulation, I encountered various issues with generating point clouds from the depth image. These are more-or-less known problems with Gazebo (Garden), but they were not resolved by applications of common methods - like introducing an optical frame joint and transform (via xacro). So I resorted to fusing the lidar and depth camera point cloud (which, if taken directly from gazebo, works fine!) in order to support rtabmap. This repo contains simple, fast tools to do this, and to decimate the depth camera point cloud data on the fly.

## point cloud fusion and decimation

See ```src/cloud_scan_fuser_node.cpp```.

## What is Gazebo's (implicit) transform from depth image to point cloud ???

Since computing the point cloud from the depth image is such a mess, one thing I attempted to do was simply compute the appropriate transform from the depth image to the point cloud. This isn't working - yet.

See ```scripts/snap_depth_cloud_and_calib.py```.

## Work in progress!
Note this is a work in progress.

## Credits

This code was developed and debugged and optimized via close collaboration with GPT5 (OpenAI, August 2025). Getting good code out of GPT5 is a ... learning process!