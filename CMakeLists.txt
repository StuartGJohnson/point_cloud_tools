cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(point_cloud_tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(pcl_conversions REQUIRED)

# This finds the PCL library itself
find_package(PCL REQUIRED COMPONENTS common filters)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# The PCL library must be found before the target link libraries are specified.
# This check ensures PCL was found successfully.
if (PCL_FOUND)
  add_executable(voxel_grid_filter_node src/voxel_grid_filter_node.cpp)
  add_executable(cloud_scan_fuser_node src/cloud_scan_fuser_node.cpp)

  # Link the executable against its dependencies
  ament_target_dependencies(voxel_grid_filter_node
    rclcpp
    sensor_msgs
    message_filters
    tf2 tf2_ros tf2_sensor_msgs
    laser_geometry
    pcl_conversions
  )

  ament_target_dependencies(cloud_scan_fuser_node
    rclcpp
    sensor_msgs
    message_filters
    tf2 tf2_ros tf2_sensor_msgs
    laser_geometry
    pcl_conversions
  )

  # Link against PCL, which is not a direct ament dependency
  target_link_libraries(voxel_grid_filter_node
    ${PCL_LIBRARIES}
  )

  target_link_libraries(cloud_scan_fuser_node
    ${PCL_LIBRARIES}
  )

  target_include_directories(voxel_grid_filter_node
    PRIVATE
    ${PCL_INCLUDE_DIRS}
  )

  target_include_directories(cloud_scan_fuser_node
    PRIVATE
    ${PCL_INCLUDE_DIRS}
  )

  # Install the executable so it can be run via `ros2 run`
  install(TARGETS
    voxel_grid_filter_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  install(TARGETS
    cloud_scan_fuser_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  add_definitions(${PCL_DEFINITIONS})
else()
  # This block will be executed if PCL is not found
  message(FATAL_ERROR "PCL not found. Please ensure it is installed and configured correctly.")
endif()

install(PROGRAMS
  scripts/snap_depth_cloud_and_calib.py
  # scripts/anything_else.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
