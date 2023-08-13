#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Relu : public Activation
  {
  public:
    double activate(double z)
    {
      return z < 0 ? 0 : z;
    };

    MatrixXd diff(MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };
  };
}
