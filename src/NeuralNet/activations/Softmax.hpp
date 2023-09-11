#pragma once

#include "interfaces/Activation.hpp"

namespace NeuralNet
{
  class Softmax : public Activation
  {
  public:
    static Eigen::MatrixXd activate(const Eigen::MatrixXd &z)
    {
      // Approach to mitigate errors resulting from exponentiating large values
      double maxValue = z.maxCoeff(); // Getting the highest value
      Eigen::MatrixXd shiftedInputs = z.array() - maxValue;
      Eigen::MatrixXd exp = shiftedInputs.array().exp();

      return exp.array() / exp.sum();
    };

    static Eigen::MatrixXd diff(const Eigen::MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };
  };
}
