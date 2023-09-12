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
      Eigen::MatrixXd scaled = scaleDown(z, 1 / z.maxCoeff());
      Eigen::MatrixXd exp = scaled.array().exp();

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
