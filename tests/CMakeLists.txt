
cmake_minimum_required(VERSION 3.5)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#import eigen
find_package(Eigen3 REQUIRED)

# Changing the cpp standard 
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(./libs/Catch2)

#create normal executable
add_executable(test-functions test-functions.cpp)

