#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Sigmoid : public Activation
  {
  public:
    static MatrixXd activate(MatrixXd &z)
    {
      MatrixXd negZ = -z;
      return 1 / (1 + negZ.array().exp());
    };

    static MatrixXd diff(MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };
  };
}
