#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

#include "Layer.hpp"

namespace NeuralNet {
class Dense : public Layer {
 public:
  Dense(int nNeurons, ACTIVATION activation = ACTIVATION::SIGMOID,
        WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0)
      : Layer(nNeurons, activation, weightInit, bias) {
    type = LayerType::DENSE;
  }

  ~Dense(){};

 private:
  // non-public serialization
  friend class cereal::access;
  Dense(){};  // Required for serialization
};
}  // namespace NeuralNet

CEREAL_REGISTER_TYPE(NeuralNet::Dense);

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNet::Layer, NeuralNet::Dense);
