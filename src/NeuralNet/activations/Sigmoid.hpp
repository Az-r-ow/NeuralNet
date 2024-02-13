#pragma once

#include "Activation.hpp"

namespace NeuralNet
{
  class Sigmoid : public Activation
  {
  public:
    static Eigen::MatrixXd activate(const Eigen::MatrixXd &z)
    {
      Eigen::MatrixXd negZ = -z;
      return 1 / (1 + negZ.array().exp());
    };

    static Eigen::MatrixXd diff(const Eigen::MatrixXd &a)
    {
      return a.array() * (1.0 - a.array());
    };
  };
}
