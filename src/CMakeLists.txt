
set(SRC_FILES 
  ${NETWORK_DIR}/Network.cpp
)

add_library(NeuralNet ${SRC_FILES})

target_include_directories(NeuralNet PUBLIC ${LIBS_DIR}/eigen ${LIBS_DIR}/ftxui/include ${LIBS_DIR}/cereal/include ${NETWORK_DIR})
target_link_directories(NeuralNet PUBLIC ${NETWORK_DIR})
target_link_libraries(NeuralNet PRIVATE ftxui::dom)