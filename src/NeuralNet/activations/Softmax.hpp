#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Softmax : public Activation
  {
  public:
    static MatrixXd activate(const MatrixXd &z)
    {
      MatrixXd exp = z.array().exp();
      double sum = exp.sum();

      return exp.array() / sum;
    };

    static MatrixXd diff(const MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };
  };
}
