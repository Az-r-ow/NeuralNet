#pragma once

#include <Eigen/Dense>

namespace NeuralNet
{
  class Activation
  {
  public:
    // Activates the input z
    static double activate(const Eigen::MatrixXd &z);
    // compute the derivative
    static Eigen::MatrixXd diff(const Eigen::MatrixXd &a);
  };
}
