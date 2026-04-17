cmake_minimum_required(VERSION 3.8)
project(mini_r1_v1_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_sim REQUIRED)

install(
  DIRECTORY launch urdf meshes worlds config params
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/left_hand_maze_nav.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME left_hand_maze_nav
)
install(
  PROGRAMS scripts/micro_mouse_wall_follower.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME micro_mouse_wall_follower
)
install(
  PROGRAMS scripts/aruco_direction_checker.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME aruco_direction_checker
)
install(
  PROGRAMS scripts/arena_navigator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME arena_navigator
)
  
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



ament_package()
