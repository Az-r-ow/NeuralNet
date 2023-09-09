#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;

namespace NeuralNet
{
  class Activation
  {
  public:
    // Activates the input z
    static double activate(MatrixXd &z);
    // compute the derivative
    static MatrixXd diff(MatrixXd &a);
  };
}
