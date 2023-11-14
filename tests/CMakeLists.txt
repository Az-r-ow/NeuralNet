cmake_minimum_required(VERSION 3.15)

function(neural_net_add_test source)
  get_filename_component(TEST_TARGET ${source} NAME_WE)
  add_executable(${TEST_TARGET} ${source})
  install(TARGETS ${TEST_TARGET} RUNTIME)
  target_link_libraries(${TEST_TARGET} PRIVATE Catch2::Catch2WithMain NeuralNet)
endfunction()

neural_net_add_test(test-activations.cpp)
neural_net_add_test(test-functions.cpp)
neural_net_add_test(test-network.cpp)
neural_net_add_test(test-tensor.cpp)