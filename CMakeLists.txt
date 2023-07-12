
cmake_minimum_required(VERSION 3.5)

# normal cmake for executables
# Set the project name
project (neurons)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#import eigen
find_package(Eigen3 REQUIRED)

# Changing the cpp standard 
set(CMAKE_CXX_STANDARD 17)

#create normal executable
add_executable(neurons main.cpp src/Neuron.cpp)

#link eigen to executables
target_link_libraries (neurons Eigen3::Eigen)
