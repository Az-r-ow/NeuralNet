#pragma once

namespace NeuralNet
{
  enum ActivationName
  {
    RELU,
    SIGMOID
  };

  enum WeightInit
  {
    RANDOM,
    GLOROT,
    HE,
    LACUN
  };
}
