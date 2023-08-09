cmake_minimum_required(VERSION 3.15)


add_executable(test-functions test-functions.cpp)

target_include_directories(test-functions PUBLIC ${SRC_DIR}/NeuralNet/utils)
target_link_directories(test-functions PUBLIC ${SRC_DIR}/NeuralNet/utils)
target_link_libraries(test-functions PRIVATE Catch2::Catch2WithMain)

add_test(test-n test-network)

add_executable(test-network test-network.cpp)

target_include_directories(test-network PUBLIC ${SRC_DIR}/NeuralNet)
target_link_directories(test-network PUBLIC ${SRC_DIR}/NeuralNet)
target_link_libraries(test-network PRIVATE Catch2::Catch2WithMain)
target_include_directories(test-network PUBLIC ${LIBS_DIR}/eigen)


add_test(test-f test-functions)