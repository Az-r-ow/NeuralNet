#pragma once

#include <cmath>
#include "Activation.hpp"

namespace NeuralNet
{
  class Softmax : public Activation
  {
  public:
    static Eigen::MatrixXd activate(const Eigen::MatrixXd &z)
    {
      Eigen::MatrixXd exp = z.array().exp();

      Eigen::MatrixXd sumExp = exp.rowwise().sum().replicate(1, exp.cols());

      return exp.array() / sumExp.array();
    };

    static Eigen::MatrixXd diff(const Eigen::MatrixXd &a)
    {
      return Eigen::MatrixXd::Constant(a.rows(), a.cols(), 1);
    };

  private:
    static Eigen::MatrixXd scale(const Eigen::MatrixXd &z, double scaleFactor)
    {
      return z * scaleFactor;
    }
  };
}
