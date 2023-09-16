#pragma once

#include "interfaces/Activation.hpp"
#include <cmath>

namespace NeuralNet
{
  class Softmax : public Activation
  {
  public:
    static Eigen::MatrixXd activate(const Eigen::MatrixXd &z)
    {
      Eigen::MatrixXd exp = z.array().exp();

      return exp / exp.sum();
    };

    static Eigen::MatrixXd diff(const Eigen::MatrixXd &a)
    {
      return a.array() * (1 - a.array());
    };

  private:
    static Eigen::MatrixXd scale(const Eigen::MatrixXd &z, double scaleFactor)
    {
      return z * scaleFactor;
    }
  };
}
