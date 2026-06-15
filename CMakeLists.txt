cmake_minimum_required(VERSION 3.5)
project(snn_aloha)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(interbotix_xs_msgs REQUIRED)
find_package(interbotix_xsarm_control REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rviz_common REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

install(
  PROGRAMS
    setup.py
    aloha_scripts/constants.py
    aloha_scripts/one_side_teleop.py
    aloha_scripts/real_env.py
    aloha_scripts/record_episodes.py
    aloha_scripts/replay_episodes.py
    aloha_scripts/robot_utils.py
    aloha_scripts/sleep.py
    aloha_scripts/visualize_episodes.py
    aloha_scripts/realsense_publisher.py
  DESTINATION
    lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
    config
    aloha2
  DESTINATION
    share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()