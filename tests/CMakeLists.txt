cmake_minimum_required(VERSION 3.15)

add_executable(test-functions test-functions.cpp)

target_include_directories(test-functions PUBLIC ${SRC_DIR}/NeuralNet/utils)
target_link_directories(test-functions PUBLIC ${SRC_DIR}/NeuralNet/utils)
target_link_libraries(test-functions PRIVATE Catch2::Catch2WithMain)

add_test(test-f test-functions)