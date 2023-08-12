#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;

namespace NeuralNet
{
  class Activation
  {
  public:
    // Activates the input z
    virtual double activate(double z) = 0;
    // compute the derivative
    virtual MatrixXd diff(MatrixXd &a) = 0;
  };
}
