#pragma once

#include "interfaces/Loss.hpp"

namespace NeuralNet
{
  /**
   * Quadratic Loss
   */
  class Quadratic : public Loss
  {
  public:
    static double cmpLoss(const Eigen::MatrixXd &o, const Eigen::MatrixXd &y)
    {
      Eigen::MatrixXd cMatrix = o.array() - y.array();
      cMatrix = cMatrix.unaryExpr(&sqr);

      return cMatrix.sum();
    };

    static Eigen::MatrixXd cmpGradient(const Eigen::MatrixXd &yHat, const Eigen::MatrixXd &y)
    {
      assert(yHat.rows() == y.rows());
      return (yHat.array() - y.array()).matrix() * 2;
    };
  };
}