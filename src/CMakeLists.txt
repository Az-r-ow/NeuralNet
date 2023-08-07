
set(SRC_FILES 
  ${SRC_DIR}/neuralNet/Network.cpp
  ${SRC_DIR}/neuralNet/Layer.cpp
)

add_library(NeuralNet ${SRC_FILES})
target_include_directories(NeuralNet PUBLIC ${LIBS_DIR}/eigen)