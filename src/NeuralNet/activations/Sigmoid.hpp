#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Sigmoid : public Activation
  {
  public:
    static double activate(double z)
    {
      return 1 / (1 + std::exp(-z));
    };

    static MatrixXd diff(MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };
  };
}
