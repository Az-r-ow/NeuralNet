cmake_minimum_required(VERSION 3.15)


add_executable(test-functions test-functions.cpp)

target_link_libraries(test-functions PRIVATE Catch2::Catch2WithMain)
target_include_directories(test-functions PUBLIC ${NETWORK_DIR}/utils)


add_test(test-n test-network)

add_executable(test-network test-network.cpp)

target_link_libraries(test-network PRIVATE Catch2::Catch2WithMain NeuralNet)
target_include_directories(test-network PUBLIC ${NETWORK_DIR})

add_test(test-f test-functions)