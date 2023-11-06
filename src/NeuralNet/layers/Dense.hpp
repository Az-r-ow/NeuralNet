#pragma once

#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include "Layer.hpp"

namespace NeuralNet
{
  class Dense : public Layer
  {
  public:
    Dense(int nNeurons, ACTIVATION activation = ACTIVATION::SIGMOID, WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0) : Layer(nNeurons, activation, weightInit, bias) {}

    ~Dense(){};

    template <class Archive>
    void serialize(Archive &ar)
    {
      ar(cereal::base_class<Layer>(this));
    };
  };
}

CEREAL_REGISTER_TYPE(NeuralNet::Dense);