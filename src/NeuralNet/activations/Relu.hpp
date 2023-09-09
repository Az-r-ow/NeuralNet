#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Relu : public Activation
  {
  public:
    static MatrixXd activate(MatrixXd &z)
    {
      return z.unaryExpr(&Relu::activateValue);
    }

    static MatrixXd diff(MatrixXd &a)
    {
      return a.unaryExpr(&Relu::diffValue);
    }

  private:
    static double diffValue(double a)
    {
      return a > 0 ? 1 : 0;
    }

    static double activateValue(double z)
    {
      return z < 0 ? 0 : z;
    }
  };
}