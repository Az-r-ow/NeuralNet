cmake_minimum_required(VERSION 3.15)

add_executable(test-functions test-functions.cpp)

target_link_libraries(test-functions PRIVATE Catch2::Catch2)

add_test(test-f test-functions)