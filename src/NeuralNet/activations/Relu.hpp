#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Relu : public Activation
  {
  public:
    static double activate(double z)
    {
      return z < 0 ? 0 : z;
    }

    static MatrixXd diff(MatrixXd &a)
    {
      MatrixXd der = a;
      return der.unaryExpr(&Relu::diffValue);
    }

  private:
    static double diffValue(double a)
    {
      return a > 0 ? 1 : 0;
    }
  };
}