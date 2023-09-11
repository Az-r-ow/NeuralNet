#pragma once

#include "interfaces/Loss.hpp"

namespace NeuralNet
{
  /**
   * Multi-class Cross Entropy
   */
  class MCE : public Loss
  {
    static double cmpLoss(const Eigen::MatrixXd &o, const Labels &y)
    {
      Eigen::MatrixXd cMatrix = y.array() * o.array().log();
      return -cMatrix.sum();
    }

    static Eigen::MatrixXd cmpGradient(const Eigen::MatrixXd &yHat, const Labels &y)
    {
      assert(yHat.rows() == y.rows());
      return yHat.array() - y.array();
    }
  };
}