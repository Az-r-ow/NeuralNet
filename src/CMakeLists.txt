
set(SRC_FILES 
  ${SRC_DIR}/NeuralNet/Network.cpp
  ${SRC_DIR}/NeuralNet/Layer.cpp
)

add_library(NeuralNet ${SRC_FILES})
target_include_directories(NeuralNet PUBLIC ${LIBS_DIR}/eigen ${NETWORK_DIR})
target_link_directories(NeuralNet PUBLIC ${NETWORK_DIR})
target_link_libraries(NeuralNet PRIVATE ftxui::dom)