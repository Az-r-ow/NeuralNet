cmake_minimum_required(VERSION 3.5)

# normal cmake for executables
# Set the project name
project ("Neural Network")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# tests options 
option(UNIT_TESTS "Testing the functions" ON)


if(UNIT_TESTS)
  enable_testing()
  add_subdirectory(libs/Catch2)
  add_subdirectory(tests)
  include(CTest)
endif()

#import eigen
find_package(Eigen3 REQUIRED)

# Changing the cpp standard 
set(CMAKE_CXX_STANDARD 17)

# adding src files 
ADD_LIBRARY(cppNeuralNetwork
  src/Network.cpp
  src/Layer.cpp
  src/Functions.cpp
)

#create normal executable
add_executable(main main.cpp)

#link eigen to executables
target_link_libraries(cppNeuralNetwork Eigen3::Eigen)
target_link_libraries (main Eigen3::Eigen)
target_link_libraries(main cppNeuralNetwork)


