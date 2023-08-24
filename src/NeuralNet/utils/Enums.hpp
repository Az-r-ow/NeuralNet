#pragma once

namespace NeuralNet
{
  enum class ActivationName
  {
    RELU,
    SIGMOID
  };

  enum class WeightInit
  {
    RANDOM,
    GLOROT,
    HE,
    LACUN
  };
}
