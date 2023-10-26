cmake_minimum_required(VERSION 3.5)

# Changing the cpp standard 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# normal cmake for executables
# Set the project name
project ("Neural Network")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# tests options 
option(UNIT_TESTS "Tests" ON)

# python bindings
option(PYBIND_BUILD "Create a python module" ON)

# Basic paths 
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NETWORK_DIR ${SRC_DIR}/NeuralNet)
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

add_subdirectory(${LIBS_DIR}/ftxui EXCLUDE_FROM_ALL)

# Set -fPIC for ftxui library targets.
set_target_properties(screen PROPERTIES
    COMPILE_OPTIONS -fPIC
)

set_target_properties(dom PROPERTIES
    COMPILE_OPTIONS -fPIC
)

add_subdirectory(src)

# Necessary for the python examples
if(PYBIND_BUILD)
  set(PY_MODULE "NeuralNetPy")
  add_subdirectory(${LIBS_DIR}/pybind11)
  pybind11_add_module(${PY_MODULE} ${SRC_DIR}/bindings/${PY_MODULE}.cpp)
  target_include_directories(${PY_MODULE} PRIVATE ${LIBS_DIR}/eigen ${NETWORK_DIR})
  target_link_directories(${PY_MODULE} PRIVATE ${LIBS_DIR}/eigen ${NETWORK_DIR})
  target_link_libraries(${PY_MODULE} PRIVATE NeuralNet)
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
target_link_directories(main PUBLIC ${LIBS_DIR}/cereal)