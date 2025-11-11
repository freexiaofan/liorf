cmake_minimum_required(VERSION 2.8.3)
project(liorf)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
# set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(Boost REQUIRED COMPONENTS timer)
find_library(LZ4_LIB lz4)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")
find_package(GeographicLib REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  message_generation
  visualization_msgs
)

add_library(nano_gicp_lib
  include/nano_gicp/nanoflann.cc
  include/nano_gicp/lsq_registration.cc
  include/nano_gicp/nano_gicp.cc
)

target_link_libraries(nano_gicp_lib
  ${PCL_LIBRARIES}
  ${OPENMP_LIBS}
)

target_include_directories(nano_gicp_lib PUBLIC
  nano_gicp/include
  ${PCL_INCLUDE_DIRS}
)

# include directories
include_directories(
  lib
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

# link directories
link_directories(
  lib
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

file(GLOB common_lib "lib/*.cpp")

###########
## Build ##
###########
# <!-- liorf_yjz_lucky_boy -->
# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp ${common_lib})
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}
        /usr/local/lib/libtbb.so
        /usr/local/lib/libtbbmalloc.so
        /usr/local/lib/libtbbmalloc_proxy.so
        )

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp ${common_lib} include/Scancontext.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization Boost::timer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} ${GeographicLib_LIBRARIES} 
        gtsam
        /usr/local/lib/libtbb.so
        /usr/local/lib/libtbbmalloc.so
        /usr/local/lib/libtbbmalloc_proxy.so
        ${LZ4_LIB}
        nano_gicp_lib
  )

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp ${common_lib})
target_link_libraries(${PROJECT_NAME}_imuPreintegration Boost::timer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} 
      gtsam
        /usr/local/lib/libtbb.so
        /usr/local/lib/libtbbmalloc.so
        /usr/local/lib/libtbbmalloc_proxy.so
        )


