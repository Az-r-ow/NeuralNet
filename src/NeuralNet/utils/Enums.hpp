#pragma once

namespace NeuralNet
{
  enum class ACTIVATION
  {
    RELU,
    SIGMOID,
    SOFTMAX,
  };

  enum class WEIGHT_INIT
  {
    RANDOM,
    GLOROT,
    HE,
    LACUN
  };

  enum class LOSS
  {
    MCE, // Multi-class Cross Entropy
    QUADRATIC
  };
}
