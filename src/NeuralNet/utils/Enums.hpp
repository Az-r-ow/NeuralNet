#pragma once

namespace NeuralNet
{
  enum class ActivationName
  {
    RELU,
    SIGMOID,
    SOFTMAX,
  };

  enum class WeightInit
  {
    RANDOM,
    GLOROT,
    HE,
    LACUN
  };
}
