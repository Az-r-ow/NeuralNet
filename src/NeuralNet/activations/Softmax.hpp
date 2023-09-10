#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Softmax : public Activation
  {
  public:
    static MatrixXd activate(const MatrixXd &z)
    {
      // Approach to mitigate errors resulting from exponentiating large values
      double maxValue = z.maxCoeff(); // Getting the highest value
      MatrixXd shiftedInputs = z.array() - maxValue;
      MatrixXd exp = shiftedInputs.array().exp();

      return exp.array() / exp.sum();
    };

    static MatrixXd diff(const MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };
  };
}
