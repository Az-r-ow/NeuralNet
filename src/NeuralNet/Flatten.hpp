#pragma once

#include <tuple>
#include "Layer.hpp"
#include "utils/Functions.hpp"

namespace NeuralNet
{
  class Flatten : public Layer
  {
  public:
    Flatten(std::tuple<int, int> inputShape, ACTIVATION activation = ACTIVATION::SIGMOID, WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0);
    ~Flatten();

  private:
    std::tuple<int, int> inputShape;

    void feedInputs(std::vector<std::vector<std::vector<double>>> inputs);
  };
}