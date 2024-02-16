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
    LECUN,
    CONSTANT // For testing
  };

  enum class LOSS
  {
    MCE, // Multi-class Cross Entropy
    QUADRATIC
  };
}
