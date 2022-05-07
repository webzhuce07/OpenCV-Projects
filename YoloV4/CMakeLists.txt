# project name
PROJECT(YoloV4)
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)

# set the directory of executable files
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Camshift_SOURCE_DIR}/bin)

# find required opencv
find_package(OpenCV REQUIRED)

# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# name of executable file and path of source file
add_executable(YoloV4 main.cpp )
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(YoloV4 ${OpenCV_LIBS})
