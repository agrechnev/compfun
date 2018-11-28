cmake_minimum_required(VERSION 3.8)
project (compfun)

set (CMAKE_CXX_STANDARD 14)


file(COPY img  DESTINATION .)

# Threads
#find_package(Threads)


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Build all examples
add_subdirectory(fun1)
add_subdirectory(fun2)
add_subdirectory(fun3)
