cmake_minimum_required(VERSION 3.8)
project( fun1 )

set (CMAKE_CXX_STANDARD 14)

add_executable( ${PROJECT_NAME} fun1.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
