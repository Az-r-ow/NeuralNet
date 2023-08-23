cmake_minimum_required(VERSION 3.5)

# normal cmake for executables
# Set the project name
project ("Neural Network")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# tests options 
option(UNIT_TESTS "Tests" OFF)

# pybind
option(PYBIND_BUILD "Create a python module" ON)

# Basic paths 
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NETWORK_DIR ${SRC_DIR}/NeuralNet)
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Changing the cpp standard 
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

if(PYBIND_BUILD)
  add_subdirectory(${LIBS_DIR}/pybind11)
  pybind11_add_module(NeuralNetPy ${SRC_DIR}/pyBindings/NeuralNet.cpp)
  target_include_directories(NeuralNetPy PUBLIC ${LIBS_DIR}/eigen ${NETWORK_DIR})
endif()

if(UNIT_TESTS)
  enable_testing()
  add_subdirectory(${LIBS_DIR}/Catch2)
  add_subdirectory(tests)
  include(CTest)
endif()

#create normal executable
add_executable(main main.cpp)

target_link_libraries(main NeuralNet)