#pragma once

#include "Loss.hpp"

namespace NeuralNet
{
  /**
   * Multi-class Cross Entropy
   */
  class MCE : public Loss
  {
  public:
    static double cmpLoss(const Eigen::MatrixXd &o, const Eigen::MatrixXd &y)
    {
      Eigen::MatrixXd cMatrix = y.array() * o.array().log();

      return -cMatrix.sum();
    };

    static Eigen::MatrixXd cmpLossGrad(const Eigen::MatrixXd &yHat, const Eigen::MatrixXd &y)
    {
      assert(yHat.rows() == y.rows() && yHat.cols() == y.cols());
      return yHat.array() - y.array();
    };
  };
}